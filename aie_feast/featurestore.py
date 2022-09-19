class FeatureStore:
    def __init__(self, project_folder=None, url=None, token=None, projectID=None):
        if project_folder:
            file_path = project_folder + r"/feature_store.yml"
        elif url and token and projectID:
            pass
        else:
            raise ValueError("one of config file or meta server project should be provided")
        # TODO: init
