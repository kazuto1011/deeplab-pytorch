from libs.datasets.cocostuff import CocoStuff10k


def get_dataset(name):
    return {
        'cocostuff': CocoStuff10k,
    }[name]
