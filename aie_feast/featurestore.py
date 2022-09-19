class FeatureStore:
    def __init__(self, project_folder=None, url=None, token=None, projectID=None):
        if project_folder:
            file_path = project_folder + r"/config.json"
        elif url and token and projectID:
            pass
        else:
            raise ValueError("config file or meta server shoule be provided")
