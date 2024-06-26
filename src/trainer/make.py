from trainer.base import MultiModalTrainer

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