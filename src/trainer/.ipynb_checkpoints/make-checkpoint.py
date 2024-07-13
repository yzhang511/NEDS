from trainer.base import MultiModalTrainer, BaselineTrainer

def make_multimodal_trainer(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    **kwargs
):
    return MultiModalTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        **kwargs
    )


def make_baseline_trainer(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    **kwargs
):
    return BaselineTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        **kwargs
    )