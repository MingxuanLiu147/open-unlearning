"""
PAI 作业 YAML 生成器

用法:
    python scripts/pai/gen_job.py --name my-experiment --cmd "python src/train.py ..."
    python scripts/pai/gen_job.py --name rome-test --editor ROME --model Qwen2.5-7B-Instruct
    python scripts/pai/gen_job.py --name grace-test --editor GRACE --model Qwen2.5-7B-Instruct --gpu 2

快捷模式 (知识编辑):
    python scripts/pai/gen_job.py --name rome-zsre --editor ROME
    python scripts/pai/gen_job.py --name memit-zsre --editor MEMIT --sku a100

自定义命令模式:
    python scripts/pai/gen_job.py --name my-job --cmd "python src/train.py --config-name=unlearn.yaml ..."
"""
import argparse
import yaml
import sys

SSH_KEY = (
    "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC0XD7JEZGiJbJe4vSAaiCOFmAQ"
    "sVz0FVfKTbifZDNZzOrGC/JavH6cT8JNinNt2JT4UFmDO1yA7pSXx6tcv21rOztI"
    "GkdOeU4xU4Q+Id3zbKq6LWDpJ/NoofoiQaHqJmQSkkZSkqSdYfj1o+Dj0S/Uht6"
    "asOzv8+BMPa8YASgYaAbf/H52rJRwUUCQwXe8KCYhFhIkoZzt/5Odz7No1vjbukw"
    "RQoPlZLZf5VLLH5JWOXI0BaYD1ubvV+hSuVl5QvoYLZvrxpWZ0JmaS/DHq5Ilciw"
    "GTceGX6Utla9Q7FoXTD4/XvphzmY4QhW+oOSaQzOGjxwEdHt4zlAjBk2K+qjesw3"
    "cdFv0/XNiQsmeDPJL5wv7p4FOia3guiddgEdoxbne7Cfv9+Ap6ey3tNTFrRDSyxB/"
    "fVhfaZeISaJuflEI8HeomEuqb39Qn0oXPNbRrf0+GI9qU1ylWUIvFvnPQx7xf6BC"
    "w0CEPaYda8NSaod64SVNI+2Vc5swhQWJnTG5E9mMhJ0bViQOU48yP43vRZVts5HZ"
    "yWlil/k0EayinHkyyTkhbVHYPxG4UpIzA6jQv3tdfX5c56cTuuKEZqxdXxHD+jq4"
    "SKNBTyraNrMFuK2xl18NQx2kKu6lTSIrF2+RrRi8Gu7gbaOfCDh4zsJS+x/24MU9"
    "REaB+7P8C4UKaD1Sbw== happy-yan"
)

STORAGE_BASE = "/mnt/confignfs/userdata/liumingxuan"
CACHE_BASE = "/mnt/confignfs/usercache/liumingxuan"

SKU_MAP = {
    "3090": {"gpu": 1, "cpu": 6, "mem": 62500, "sku": "gpu-machine-3090"},
    "a100": {"gpu": 4, "cpu": 120, "mem": 500000, "sku": "gpu-machine-a100-lt"},
}

EDITORS = [
    "ROME", "MEMIT", "MEND", "GRACE", "WISE", "IKE",
    "AlphaEdit", "UNKE", "SERAC", "MALMEN", "InstructEdit", "AnyEdit",
]


def build_edit_commands(editor, model, task_name):
    model_path = f"{STORAGE_BASE}/models/{model}"
    return [
        f"export HF_HOME={CACHE_BASE}/huggingface",
        f"export TRANSFORMERS_CACHE={CACHE_BASE}/huggingface",
        f"export HF_DATASETS_CACHE={CACHE_BASE}/huggingface/datasets",
        "cd /workspace/open-unlearning",
        (
            f"python src/train.py --config-name=edit.yaml"
            f" experiment=edit/zsre/default"
            f" trainer=edit/{editor}"
            f" task_name={task_name}"
            f" model.model_args.pretrained_model_name_or_path={model_path}"
            f" model.tokenizer_args.pretrained_model_name_or_path={model_path}"
        ),
        'echo "=== Job Finished ==="',
    ]


def build_custom_commands(cmd):
    return [
        f"export HF_HOME={CACHE_BASE}/huggingface",
        f"export TRANSFORMERS_CACHE={CACHE_BASE}/huggingface",
        "cd /workspace/open-unlearning",
        cmd,
        'echo "=== Job Finished ==="',
    ]


def gen_yaml(args):
    sku = SKU_MAP[args.sku]
    gpu_count = args.gpu if args.gpu else sku["gpu"]

    if args.editor:
        model = args.model or "Qwen2.5-7B-Instruct"
        task_name = args.name
        commands = build_edit_commands(args.editor, model, task_name)
    elif args.cmd:
        commands = build_custom_commands(args.cmd)
    else:
        print("错误: 必须指定 --editor 或 --cmd", file=sys.stderr)
        sys.exit(1)

    job = {
        "protocolVersion": 2,
        "name": args.name,
        "type": "job",
        "jobRetryCount": 0,
        "prerequisites": [{
            "type": "dockerimage",
            "uri": f"210.75.240.150:30003/liumingxuan/open-unlearning:{args.tag}",
            "name": "docker_image0",
        }],
        "taskRoles": {
            "taskrole": {
                "instances": 1,
                "completion": {"minFailedInstances": 1},
                "taskRetryCount": 0,
                "dockerImage": "docker_image0",
                "resourcePerInstance": {
                    "gpu": gpu_count,
                    "cpu": sku["cpu"],
                    "memoryMB": sku["mem"],
                },
                "commands": commands,
            }
        },
        "defaults": {"virtualCluster": args.vc},
        "extras": {
            "com.microsoft.pai.runtimeplugin": [
                {
                    "plugin": "ssh",
                    "parameters": {
                        "jobssh": True,
                        "userssh": {"type": "custom", "value": SSH_KEY},
                    },
                },
                {
                    "plugin": "teamwise_storage",
                    "parameters": {
                        "storageConfigNames": ["usercache", "userdata"],
                    },
                },
            ]
        },
        "hivedScheduler": {
            "taskRoles": {
                "taskrole": {
                    "skuNum": gpu_count,
                    "skuType": sku["sku"],
                }
            }
        },
    }
    return job


def main():
    p = argparse.ArgumentParser(description="PAI 作业 YAML 生成器")
    p.add_argument("--name", required=True, help="作业名称")
    p.add_argument("--editor", choices=EDITORS, help="知识编辑方法 (快捷模式)")
    p.add_argument("--model", default="Qwen2.5-7B-Instruct", help="模型名 (需已同步到 netdisk)")
    p.add_argument("--cmd", help="自定义命令 (自定义模式)")
    p.add_argument("--sku", default="3090", choices=SKU_MAP.keys(), help="GPU 类型")
    p.add_argument("--gpu", type=int, help="GPU 数量 (覆盖 SKU 默认值)")
    p.add_argument("--vc", default="default", help="Virtual Cluster")
    p.add_argument("--tag", default="v1", help="Docker 镜像 tag")
    p.add_argument("-o", "--output", help="输出文件 (默认打印到 stdout)")
    args = p.parse_args()

    job = gen_yaml(args)
    text = yaml.dump(job, default_flow_style=False, allow_unicode=True, sort_keys=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"已生成: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
