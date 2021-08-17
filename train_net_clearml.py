#!/usr/bin/env python

if __name__ == "__main__":
    import os
    from pathlib import Path

    from clearml import Task

    from det2_default_argparser import default_argument_parser

    parser = default_argument_parser()
    parser.add_argument("--skip-clearml", help='flag to entirely skip any clearml action.', action='store_true')
    parser.add_argument("--monitor-ps", help='flag to monitoring processes.', action='store_true')
    parser.add_argument("--clearml-run-locally", help='flag to run job locally but keep clearml expt tracking.', action='store_true')
    ## CLEARML ARGS
    parser.add_argument("--clearml-proj", default="det2", help="ClearML Project Name")
    parser.add_argument("--clearml-task-name", default="Task", help="ClearML Task Name")
    parser.add_argument("--clearml-task-type", default="data_processing", help="ClearML Task Type, e.g. training, testing, inference, etc", choices=['training','testing','inference','data_processing','application','monitor','controller','optimizer','service','qc','custom'])
    parser.add_argument("--docker-img", default="harbor.dsta.ai/nvidia/pytorch:21.03-py3", help="Base docker image to pull")
    parser.add_argument("--queue", default="1gpu", help="ClearML Queue")
    ### S3
    parser.add_argument("--skip-s3", help='flag to entirely skip any s3 action.', action='store_true')
    ## DOWNLOAD MODELS ARGS
    parser.add_argument(
        "--download-models",
        help="List of models to download",
        nargs='+'
    )
    parser.add_argument("--s3-models-bucket", help="S3 Bucket for models")
    parser.add_argument("--s3-models-path", help="S3 Models Path")
    ## Model weights to load for training
    parser.add_argument(
        "--model-weights", 
        help="MODEL.WEIGHTS | Path to pretrained model weights")
    ## DOWNLOAD DATA ARGS
    parser.add_argument(
        "--download-data",
        help="List of dataset to download",
        nargs='+'
    )
    parser.add_argument("--s3-data-bucket", help="S3 Bucket for data")
    parser.add_argument("--s3-data-path", help="S3 Data Path")
    parser.add_argument(
        "--custom-dsnames",
        help='Names of custom datasets (must match to those in args.datasets_train and args.datasets_test). Only for custom datasets that does not conform to repo dataset assumptions.',
        nargs='*'
        )
    parser.add_argument(
        "--custom-cocojsons",
        help='Paths to coco json file. Only for custom datasets that does not conform to repo dataset assumptions.',
        nargs='*'
        )
    parser.add_argument(
        "--custom-imgroots",
        help='Paths to img roots. Only for custom datasets that does not conform to repo dataset assumptions.',
        nargs='*'
        )
    # Datasets to register, will override config.yaml
    parser.add_argument(
        "--datasets-train",
        help="DATASETS.TRAIN"
    )
    parser.add_argument(
        "--datasets-test",
        help="DATASETS.TEST"
    )
    ## UPLOAD OUTPUT ARGS
    parser.add_argument("--s3-output-bucket", help="S3 Bucket for output")
    parser.add_argument("--s3-output-path", help="S3 Path to output")
    ## Hyperparams
    parser.add_argument(
        "--test-eval-period",
        help="TEST.EVAL_PERIOD"
    )
    parser.add_argument(
        "--solver-ims-per-batch",
        help="SOLVER.IMS_PER_BATCH"
    )
    parser.add_argument(
        "--solver-base-lr",
        help="SOLVER.BASE_LR"
    )
    parser.add_argument(
        "--solver-warmup-iters",
        help="SOLVER.WARMUP_ITERS"
    )
    parser.add_argument(
        "--solver-steps",
        help="SOLVER.STEPS"
    )
    parser.add_argument(
        "--solver-max-iter",
        help="SOLVER.MAX_ITER"
    )
    parser.add_argument(
        "--dataloader-num-workers",
        help="DATALOADER.NUM_WORKERS"
    )

    args = parser.parse_args()
    print("Command Line Args:", args)

    AWS_ENDPOINT_URL=os.environ.get("AWS_ENDPOINT_URL", "https://ecs.dsta.ai")
    AWS_ACCESS_KEY=os.environ.get("AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS=os.environ.get("AWS_SECRET_ACCESS")
    '''
    clearml task init
    '''
    if not args.skip_clearml:
        cl_task = Task.init(project_name=args.clearml_proj,task_name=args.clearml_task_name, task_type=args.clearml_task_type)
        cl_task.set_base_docker(f"{args.docker_img} --env GIT_SSL_NO_VERIFY=true --env AWS_ACCESS_KEY={AWS_ACCESS_KEY} --env AWS_SECRET_ACCESS={AWS_SECRET_ACCESS}")
        if not args.clearml_run_locally:
            cl_task.execute_remotely(queue_name=args.queue, exit_process=True)

    else:
        cl_task = None

    '''
    S3 handling to download weights and datasets
    '''    
    from utils.s3_helper import S3_handler
    import ssl
    import wget

    if not args.skip_s3:
        CERT_PATH=os.environ.get("CERT_PATH", "/usr/share/ca-certificates/extra/ca.dsta.ai.crt")
        CERT_DL_URL='http://gitlab.dsta.ai/ai-platform/getting-started/raw/master/config/ca.dsta.ai.crt'
        if CERT_DL_URL:
            ssl._create_default_https_context = ssl._create_unverified_context
            wget.download(CERT_DL_URL)
            CERT_PATH = Path(CERT_DL_URL).name

        s3_handler = S3_handler(AWS_ENDPOINT_URL, AWS_ACCESS_KEY, AWS_SECRET_ACCESS, CERT_PATH)

        local_weight_dir = 'weights'
        local_data_dir = 'datasets'
        local_output_dir = 'output'

        local_weights_paths= s3_handler.dl_files(args.download_models, args.s3_models_bucket, args.s3_models_path, local_weight_dir, unzip=True)

        local_data_dirs = s3_handler.dl_dirs(args.download_data, args.s3_data_bucket, args.s3_data_path, local_data_dir, unzip=True)

    '''
    Datasets Registration
    '''
    from utils.det2_helper import register_datasets, parse_datasets_args, extend_opts

    # Register the custom datasets that don't conform to dataset format assumptions first
    # already_reged = []
    # if args.custom_dsnames:
    #     assert len(args.custom_dsnames)==len(args.custom_cocojsons)
    #     assert len(args.custom_dsnames)==len(args.custom_imgroots)
    #     for dsname, cjson, imroot in zip(args.custom_dsnames, args.custom_cocojsons, args.custom_imgroots):
    #         register_datasets(dsname, json_path=cjson, dataset_image_root=imroot)
    #         already_reged.append(dsname)
        # register_coco_instances("coco_train", {}, '/media/dh/HDD/coco/annotations/instances_train2017.json', '/media/dh/HDD/coco/train2017')
        # register_coco_instances("coco_val", {}, '//media/dh/HDD/coco/smallval/instances_val2017.json', '/media/dh/HDD/coco/smallval/images')
    # Then register remaining of train and test sets, assuming remainders all conform to dataset format.
    datasets_to_reg = []
    datasets_train = parse_datasets_args(args.datasets_train, datasets_to_reg)
    datasets_test = parse_datasets_args(args.datasets_test, datasets_to_reg)
    # remainder_sets = list(set(datasets_to_reg)-set(already_reged))
    # for dataset_to_reg in remainder_sets:
    #     register_datasets(dataset_to_reg, local_data_dir=local_data_dir)

    extend_opts(args.opts, 'DATASETS.TRAIN', datasets_train)
    extend_opts(args.opts, 'DATASETS.TEST', datasets_test)
    extend_opts(args.opts, 'MODEL.WEIGHTS', args.model_weights)

    extend_opts(args.opts, 'TEST.EVAL_PERIOD', args.test_eval_period)
    extend_opts(args.opts, 'SOLVER.IMS_PER_BATCH', args.solver_ims_per_batch)
    extend_opts(args.opts, 'SOLVER.BASE_LR', args.solver_base_lr)
    extend_opts(args.opts, 'SOLVER.WARMUP_ITERS', args.solver_warmup_iters)
    extend_opts(args.opts, 'SOLVER.STEPS', args.solver_steps)
    extend_opts(args.opts, 'SOLVER.MAX_ITER', args.solver_max_iter)
    extend_opts(args.opts, 'DATALOADER.NUM_WORKERS', args.dataloader_num_workers)

    '''
    Launching detectron2 run
    '''
    from detectron2.engine import launch

    from trainer import main

    if args.monitor_ps:
        from utils.psutil_helper import start_monitor
        start_monitor(freq=1)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        # args=(args, cl_task),
    )

    '''
    S3 handling to upload outputs
    '''
    if not args.skip_s3:
        s3_handler.ul_dir(local_output_dir, args.s3_output_bucket, args.s3_output_path, f'{args.clearml_task_name}')
