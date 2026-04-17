import os
import argparse
import random

import torch
import torchvision
import torchvision.transforms as transforms

from models_torch import load_model
from configs import bcolors
from utils_torch import (
    save_image,
    neuron_covered,
    init_coverage_table_single,
    get_feature_num_neurons,
    update_coverage_from_features,
    neuron_to_cover,
    normalize_gradient,
    constraint_light,
    constraint_occl,
    constraint_black,
    get_argmax_label,
    disagreement_found,
    make_objective,
)


def main():
    parser = argparse.ArgumentParser(
        description="Difference-inducing input generation for CIFAR-10 (PyTorch DeepXplore-style)"
    )
    parser.add_argument("transformation", choices=["light", "occl", "blackout"])
    parser.add_argument("weight_diff", type=float)
    parser.add_argument("weight_nc", type=float)
    parser.add_argument("step", type=float)
    parser.add_argument("seeds", type=int)
    parser.add_argument("grad_iterations", type=int)
    parser.add_argument("threshold", type=float)

    parser.add_argument("-t", "--target_model", choices=[0, 1, 2], default=0, type=int)
    parser.add_argument("--start_x", type=int, default=0)
    parser.add_argument("--start_y", type=int, default=0)
    parser.add_argument("--occl_w", type=int, default=8)
    parser.add_argument("--occl_h", type=int, default=8)

    parser.add_argument("--model1_path", type=str, default="./checkpoints/model1.pth")
    parser.add_argument("--model2_path", type=str, default="./checkpoints/model2.pth")
    parser.add_argument("--model3_path", type=str, default="./checkpoints/model3.pth")
    parser.add_argument("--output_dir", type=str, default="./generated_inputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    model1 = load_model(args.model1_path, device)
    model2 = load_model(args.model2_path, device)
    model3 = load_model(args.model3_path, device)

    # coverage table 초기화용 dummy forward
    dummy_img, _ = testset[0]
    dummy_img = dummy_img.unsqueeze(0).to(device)

    with torch.no_grad():
        _, feat1 = model1.forward_with_features(dummy_img)
        _, feat2 = model2.forward_with_features(dummy_img)
        _, feat3 = model3.forward_with_features(dummy_img)

    layer_info1 = get_feature_num_neurons(feat1)
    layer_info2 = get_feature_num_neurons(feat2)
    layer_info3 = get_feature_num_neurons(feat3)

    model_layer_dict1 = init_coverage_table_single("model1", layer_info1)
    model_layer_dict2 = init_coverage_table_single("model2", layer_info2)
    model_layer_dict3 = init_coverage_table_single("model3", layer_info3)

    print(bcolors.OKBLUE + "Start generating difference-inducing inputs..." + bcolors.ENDC)

    found_count = 0

    for seed_idx in range(args.seeds):
        img, _ = random.choice(testset)
        gen_img = img.unsqueeze(0).to(device)
        orig_img = gen_img.clone().detach()

        with torch.no_grad():
            logits1, feat1 = model1.forward_with_features(gen_img)
            logits2, feat2 = model2.forward_with_features(gen_img)
            logits3, feat3 = model3.forward_with_features(gen_img)

        label1 = get_argmax_label(logits1)
        label2 = get_argmax_label(logits2)
        label3 = get_argmax_label(logits3)

        if disagreement_found([label1, label2, label3]):
            print(
                bcolors.OKGREEN +
                f"[seed {seed_idx}] already different: {label1}, {label2}, {label3}" +
                bcolors.ENDC
            )

            update_coverage_from_features(feat1, "model1", model_layer_dict1, args.threshold)
            update_coverage_from_features(feat2, "model2", model_layer_dict2, args.threshold)
            update_coverage_from_features(feat3, "model3", model_layer_dict3, args.threshold)

            nc1 = neuron_covered(model_layer_dict1)
            nc2 = neuron_covered(model_layer_dict2)
            nc3 = neuron_covered(model_layer_dict3)
            averaged_nc = (nc1[0] + nc2[0] + nc3[0]) / float(nc1[1] + nc2[1] + nc3[1])

            print(
                bcolors.OKGREEN +
                "covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f"
                % (len(model_layer_dict1), nc1[2], len(model_layer_dict2), nc2[2], len(model_layer_dict3), nc3[2]) +
                bcolors.ENDC
            )
            print(bcolors.OKGREEN + "averaged covered neurons %.3f" % averaged_nc + bcolors.ENDC)

            save_path = os.path.join(
                args.output_dir,
                f"already_differ_{label1}_{label2}_{label3}_{seed_idx}.png"
            )
            save_image(save_path, gen_img)
            found_count += 1
            continue

        orig_label = label1

        for iters in range(args.grad_iterations):
            gen_img.requires_grad_(True)

            logits1, feat1 = model1.forward_with_features(gen_img)
            logits2, feat2 = model2.forward_with_features(gen_img)
            logits3, feat3 = model3.forward_with_features(gen_img)

            n1 = neuron_to_cover(model_layer_dict1)
            n2 = neuron_to_cover(model_layer_dict2)
            n3 = neuron_to_cover(model_layer_dict3)

            objective = make_objective(
                logits_list=[logits1, logits2, logits3],
                orig_label=orig_label,
                target_model=args.target_model,
                weight_diff=args.weight_diff,
                feature_dicts=[feat1, feat2, feat3],
                target_neurons=[n1, n2, n3],
                weight_nc=args.weight_nc,
            )

            model1.zero_grad()
            model2.zero_grad()
            model3.zero_grad()

            if gen_img.grad is not None:
                gen_img.grad.zero_()

            objective.backward()
            grads = normalize_gradient(gen_img.grad.detach())

            if args.transformation == "light":
                grads = constraint_light(grads)
            elif args.transformation == "occl":
                grads = constraint_occl(
                    grads,
                    start_point=(args.start_x, args.start_y),
                    occlusion_size=(args.occl_w, args.occl_h),
                )
            elif args.transformation == "blackout":
                grads = constraint_black(grads)

            gen_img = gen_img.detach() + grads * args.step
            gen_img = torch.clamp(gen_img, 0.0, 1.0)

            with torch.no_grad():
                pred1 = get_argmax_label(model1(gen_img))
                pred2 = get_argmax_label(model2(gen_img))
                pred3 = get_argmax_label(model3(gen_img))

            if disagreement_found([pred1, pred2, pred3]):
                with torch.no_grad():
                    _, feat1 = model1.forward_with_features(gen_img)
                    _, feat2 = model2.forward_with_features(gen_img)
                    _, feat3 = model3.forward_with_features(gen_img)

                update_coverage_from_features(feat1, "model1", model_layer_dict1, args.threshold)
                update_coverage_from_features(feat2, "model2", model_layer_dict2, args.threshold)
                update_coverage_from_features(feat3, "model3", model_layer_dict3, args.threshold)

                nc1 = neuron_covered(model_layer_dict1)
                nc2 = neuron_covered(model_layer_dict2)
                nc3 = neuron_covered(model_layer_dict3)
                averaged_nc = (nc1[0] + nc2[0] + nc3[0]) / float(nc1[1] + nc2[1] + nc3[1])

                print(
                    bcolors.OKGREEN +
                    f"[seed {seed_idx}, iter {iters}] found disagreement: {pred1}, {pred2}, {pred3}" +
                    bcolors.ENDC
                )
                print(
                    bcolors.OKGREEN +
                    "covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f"
                    % (len(model_layer_dict1), nc1[2], len(model_layer_dict2), nc2[2], len(model_layer_dict3), nc3[2]) +
                    bcolors.ENDC
                )
                print(bcolors.OKGREEN + "averaged covered neurons %.3f" % averaged_nc + bcolors.ENDC)

                base = f"{args.transformation}_{pred1}_{pred2}_{pred3}_{seed_idx}_{iters}"
                save_image(os.path.join(args.output_dir, base + ".png"), gen_img)
                save_image(os.path.join(args.output_dir, base + "_orig.png"), orig_img)

                found_count += 1
                break

    print(bcolors.OKBLUE + f"Finished. Found {found_count} disagreement-inducing inputs." + bcolors.ENDC)


if __name__ == "__main__":
    main()