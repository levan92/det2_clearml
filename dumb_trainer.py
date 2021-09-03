import detectron2.utils.comm as comm

from clearml import Task

from torch.utils.tensorboard import SummaryWriter

def main(args, cl_proj_name=None, cl_task_name=None, cl_task_id=None):
    if comm.is_main_process() and cl_task_id is not None:
        # cl_task = Task.init(project_name=cl_proj_name,task_name=cl_task_name)
        cl_task = Task.current_task()
        assert cl_task.task_id == cl_task_id,'Current task in process does not match given task id!'
        print(f'Connected clearml task: {cl_task.task_id}!')
    else:
        cl_task = None

    if comm.is_main_process():
        cl_logger = cl_task.get_logger()
        cl_logger.report_scalar("test logger","series", value=0, iteration=0)
        cl_logger.report_scalar("test logger","series", value=1, iteration=50)
        cl_logger.report_scalar("test logger","series", value=2, iteration=100)

    if comm.is_main_process():
        writer = SummaryWriter('output2')
        writer.add_scalar('metric', 0, 0)
        writer.add_scalar('metric', 0.5, 50)
        writer.add_scalar('metric', 1.0, 100)
        print('FOR CL DEBUG: wrote to TB')

        writer.close()

    return 