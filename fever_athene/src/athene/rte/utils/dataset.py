from csv import DictReader


class DataSet:
    """
    Define class for Fake News Challenge data
    """

    def __init__(self, file_stances, file_bodies):

        # Load data
        self.instances = self.read(file_stances)
        bodies = self.read(file_bodies)
        self.heads = {}
        self.bodies = {}

        # Process instances
        for instance in self.instances:
            if instance['Claim'] not in self.heads:
                head_id = len(self.heads)
                self.heads[instance['Claim']] = head_id
            instance['Body ID'] = int(instance['Body ID'])

        # Process bodies
        for body in bodies:
            self.bodies[int(body['Body ID'])] = body['Snippets']

    def read(self, filename):

        """
        Read Fake News Challenge data from CSV file
        Args:
            filename: str, filename + extension
        Returns:
            rows: list, of dict per instance
        """

        # Initialise
        rows = []

        # Process file
        with open(filename, "r", encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)

        return rows
