from gdeep.data.graph_dataloaders import create_dataloaders

def test_create_dataloaders_PROTEINS():
    train_loader, test_loader = create_dataloaders("PROTEINS", batch_size=3)
    assert len(train_loader) == 297
    assert len(test_loader) == 75
    
    batch = next(iter(train_loader))
    assert len(batch) == 3
    assert batch[0].shape == (3, 149, 6)
    assert batch[1].shape == (3, 149, 6)
    assert batch[2].shape == (3,)
    