from dateutil.relativedelta import relativedelta
from datetime import datetime

from clusterization import DailyClusters
from communities import CommunitiesClustering
from utils.data_prep import (
    CategoriesDistanceIngestion,
    ClusterizedDocumentsTable,
    ClustersTable,
)


def main() -> None:
    # tables
    clusters_table = ClustersTable()
    clusters_table.init(reset=False)

    clusterized_documents_table = ClusterizedDocumentsTable()
    clusterized_documents_table.init(reset=False)

    ref_day: datetime = datetime.now() - relativedelta(days=1)

    # cluster creation
    dc = DailyClusters(
        ref_day=ref_day,
        log_to_file=False,
    )
    dc.run()

    # communities creation
    cc = CommunitiesClustering(
        ref_day=ref_day,
        log_to_file=False,
    )
    cc.run()


if __name__ == "__main__":
    main()
