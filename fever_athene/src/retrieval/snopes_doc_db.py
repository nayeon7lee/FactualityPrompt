import json


class SnopesDocDB(object):
    def __init__(self, db_path: str):
        self.path = db_path
        with open(self.path) as f:
            self.db_dict = json.load(f)

    def path(self):
        return self.path

    def get_doc_ids(self):
        results = list(self.db_dict.keys())
        return results

    def get_doc_text(self, doc_id):
        return self.get_doc_lines(doc_id)

    def get_doc_lines(self, doc_id):
        if doc_id not in self.db_dict:
            return None
        lines = [str(num) + '\t' + line for num, line in enumerate(self.db_dict[doc_id])]
        return '\n'.join(lines)

    def get_non_empty_doc_ids(self):
        return [result for result in self.get_doc_ids() if len(self.db_dict[result]) > 0]
