# from ast import Param
from drqa.retriever import DocDB, utils


class FeverDocDB(DocDB):

    def __init__(self,path=None):
        super().__init__(path)

    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?",
            (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_non_empty_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        # cursor.execute("SELECT id FROM documents WHERE length(trim(text)) > 0")
        cursor.execute("SELECT id FROM documents WHERE length(trim(lines)) > 0")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results



def main():
    print("hi?")
    db = FeverDocDB(path = "/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/db/fever.db")
    # lines = db.get_doc_lines("Lorelai_Gilmore")
    lines = db.get_doc_lines("Goalkeeper_(association_football)")
    print(lines)

    

    # db = FeverDocDB(path = "/gpfs/fs1/projects/gpu_adlr/datasets/nayeonl/db/kilt_db.db")
    # lines = db.get_doc_lines('Michael Jordan')
    # print(lines)


if __name__ == '__main__':
    main()

