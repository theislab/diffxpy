import numpy as np
import pandas as pd

from ..stats import stats
from ..testing import correction
from ..testing.base import _DifferentialExpressionTest


class RefSets():
    """
    Class for a list of gene sets.

    Input:
    1. Read gene sets from file.
    2. Give a list with gene sets.
    3. Manually add gene sets one by one.

    .gmt files can be downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp for example.
    """

    class _Set():
        """ 
        Class for a single gene set.
        """

        def __init__(self, id: str, source: str, gene_ids: list):
            self.id = id
            self.source = source
            self.genes = set(gene_ids)
            self.intersect = None
            self.len = len(self.genes)

        def clean(self, ids):
            """ Only keep gene ids that are contained within a full 
            reference set of ids.
            """
            self.genes.intersection_update(ids)
            self.len = len(self.genes)

    def __init__(self, sets=None, fn=None, type='gmt'):
        if sets is not None:
            self.load_sets(sets, type=type)
            self._genes = np.sort(np.unique(np.concatenate([np.asarray(list(x.genes)) for x in self.sets])))
        elif fn is not None:
            self.read_from_file(fn=fn, type=type)
            self._genes = np.sort(np.unique(np.concatenate([np.asarray(list(x.genes)) for x in self.sets])))
        else:
            self.sets = []
            self._genes = np.array([])
        self._ids = [x.id for x in self.sets]
        self._set_lens = np.array([x.len for x in self.sets])
        self.genes_discarded = None

    ## Input functions.

    def load_sets(self, sets, type='gmt'):
        """ 
        Load gene sets from python list.
        """
        if type == 'gmt':
            self._load_as_gmt(sets)
        elif type == 'refset':
            self._load_as_refset(sets)
        else:
            raise ValueError('type not recognized in RefSets.load_sets()')

    def _load_as_gmt(self, sets):
        """ 
        Load gene sets from python list formatted like .gmt files.

        Use .gmt convention: sets is a list of sets
        where each list corresponds to one gene set.
        The first entry of each set is its identifier,
        the second entry is its source (e.g. a website)
        and the the thrid to last entry are the gene 
        identifiers of this set.
        """
        self.sets = [self._Set(id=x[0], source=x[1], gene_ids=x[2:]) for x in sets]

    def _load_as_refset(self, sets):
        """ 
        Load gene sets from RefSet instance.
        """
        self.sets = [self._Set(id=x.id, source=x.source, gene_ids=x.genes) for x in sets]

    def read_from_file(self, fn, type='gmt'):
        """ 
        Process gene sets from file.
        """
        if type == 'gmt':
            self._read_from_gmt(fn)
        else:
            raise ValueError('file type not recognized in RefSets.read_from_file()')

    def _read_from_gmt(self, fn):
        """ 
        Process gene sets from .gmt file.
        """
        with open(fn) as f:
            sets_raw = f.readlines()
        sets_proc = [x.split('\n')[0] for x in sets_raw]
        sets_proc = [x.split('\t') for x in sets_proc]
        sets_proc = [self._Set(id=x[0], source=x[1], gene_ids=x[2:]) for x in sets_proc]
        self.sets = sets_proc

    def add(self, id: str, source: str, gene_ids: list):
        """ 
        Add a gene set manually.
        """
        self.sets.append(self._Set(id=id, source=source, gene_ids=gene_ids))
        # Update summary variables:
        self._genes = np.sort(np.unique(np.concatenate([np.asarray(list(x.genes)) for x in self.sets])))
        self._ids = [x.id for x in self.sets]
        self._set_lens = np.array([x.len for x in self.sets])

    ## Processing functions.

    def clean(self, ids):
        """ 
        Only keep gene ids that are contained within a full 
        reference set of ids.
        """
        gene_ids_before = set(self._genes)
        for x in self.sets:
            x.clean(ids)
        gene_ids_after = set(self._genes)
        self.genes_discarded = np.asarray(list(gene_ids_before.difference(gene_ids_after)))

    def subset(self, idx: np.array):
        """ 
        Subset RefSets object.

        :param idx: np.array
            Indices of gene sets to keep.
        """
        return RefSets(sets=[self.sets[i] for i in idx], type='refset')

    def subset_bykey(self, keys: list):
        """ 
        Only keep sets that are contain at least one of a list of key strings
        in their identifier.

        :param keys: list
            List of substrings of gene set identifiers. Only gene
            sets with identifier which contains at least one key
            are returned.
        """
        idx = np.where([any([key in x for key in keys]) for x in self._ids])[0]
        print(str(len(idx)) + ' out of ' + str(len(self._ids)) + ' gene sets were kept.')
        return self.subset(idx=idx)

    ## Accession functions.

    def grepv_sets(self, x):
        """ 
        Search gene set identifiers for a substring. 
        """
        hits = np.where([any([xx in set_id for xx in x]) for set_id in self._ids])[0]
        return [self._ids[i] for i in hits]

    def get_set(self, id):
        """ 
        Return the set with a given set identifier.
        """
        return self.sets[self._ids.index(id)]

    ## Overlap functions.

    def overlap(self, enq_set: set, set_id=None):
        """ 
        Count number of overlapping genes between an internal sets and 
        a reference set.

        :param enq_set: 
            Set which contains the gene identifiers of a given enquiry set.
        :param set_id:
            Name of a specific set in self.sets against
            which the reference sets are to be overlapped.
            All sef.sets are chosen if set_id is None.
        """
        if set_id is None:
            for x in self.sets:
                x.intersect = x.genes.intersection(enq_set)
        else:
            x.intersect = self.get_set(id).genes.intersection(enq_set)


def test(
        RefSets: RefSets,
        DETest: _DifferentialExpressionTest = None,
        pval: np.array = None,
        gene_ids: list = None,
        de_threshold=0.05,
        all_ids=None,
        clean_ref=True,
        upper=False
):
    """ Perform gene set enrichment.

    Wrapper for Enrich. Just wrote this so that Enrich shows up with a
    nice doc string and that the call to this is de.enrich.test which
    makes more sense to me than de.enrich.Enrich.

    :param RefSets:
        The annotated gene sets against which enrichment is tested.
    :param DETest:
        The differential expression results object which is tested
        for enrichment in the gene sets.
    :param pval:
        Alternative to DETest, vector of p-values for differential expression.
    :param gene_ids:
        If pval was supplied instead of DETest, use gene_ids to supply the
        vector of gene identifiers (strings) that correspond to the p-values
        which can be matched against the identifieres in the sets in RefSets.
    :param de_threshold:
        Significance threshold at which a differential test (a multiple-testing 
        corrected p-value) is called siginficant. This 
    :param all_ids:
        Set of all gene identifiers, this is used as the background set in the
        hypergeometric test. Only supply this if not all genes were tested
        and are supplied above in DETest or gene_ids.
    :param clean_ref:
        Whether or not to only retain gene identifiers in RefSets that occur in 
        the background set of identifiers supplied here through all_ids.
    :param upper:
        Make all gene IDs captial.
    """
    return Enrich(
        RefSets=RefSets,
        DETest=DETest,
        pval=pval,
        gene_ids=gene_ids,
        de_threshold=de_threshold,
        all_ids=all_ids,
        clean_ref=clean_ref,
        upper=upper)


class Enrich():
    """
    """

    def __init__(
            self,
            RefSets: RefSets,
            DETest: _DifferentialExpressionTest = None,
            pval: np.array = None,
            gene_ids: list = None,
            de_threshold=0.05,
            all_ids=None,
            clean_ref=True,
            upper=False
    ):
        self._n_overlaps = None
        self._pval_enrich = None
        self._qval_enrich = None
        # Load multiple-testing-corrected differential expression
        # p-values from differential expression output.
        if DETest is not None:
            self._qval_de = DETest.qval
            self._gene_ids = DETest.gene_ids
        elif pval is not None and gene_ids is not None:
            self._qval_de = np.asarray(pval)
            self._gene_ids = gene_ids
        else:
            raise ValueError('Supply either DETest or pval and gene_ids to Enrich().')
        # Take out NA genes labels:
        # Select significant genes based on user defined threshold.
        if any([x is np.nan for x in self._gene_ids]):
            idx_notnan = np.where([x is not np.nan for x in self._gene_ids])[0]
            print('Discarded ' + str(len(self._gene_ids) - len(idx_notnan)) + ' nan gene ids, leaving ' +
                  str(len(idx_notnan)) + ' genes.')
            self._qval_de = self._qval_de[idx_notnan]
            self._gene_ids = self._gene_ids[idx_notnan]

        self._significant_de = self._qval_de <= de_threshold
        self._significant_ids = set(self._gene_ids[np.where(self._significant_de)[0]])
        if all_ids is not None:
            self._all_ids = set(all_ids)
        else:
            self._all_ids = set(self._gene_ids)

        if upper == True:
            self._gene_ids = [x.upper() for x in self._gene_ids]
            self._all_ids = set([x.upper() for x in self._all_ids])

        # Generate diagnostic statistic of number of possible overlaps in total.
        print(str(len(set(self._all_ids).intersection(set(RefSets._genes)))) +
              ' overlaps found between refset (' + str(len(RefSets._genes)) +
              ') and provided gene list (' + str(len(self._all_ids)) + ').')
        self.missing_genes = list(set(RefSets._genes).difference(set(self._all_ids)))
        # Clean reference set to only contains ids that were observed in
        # current study if required.
        self.RefSets = RefSets
        if clean_ref == True:
            self.RefSets.clean(self._all_ids)
        # Print if there are empty sets.
        idx_nonempty = np.where([len(x.genes) > 0 for x in self.RefSets.sets])[0]
        if len(self.RefSets.sets) - len(idx_nonempty) > 0:
            print('Found ' + str(len(self.RefSets.sets) - len(idx_nonempty)) +
                  ' empty sets, removing those.')
            self.RefSets = self.RefSets.subset(idx=idx_nonempty)
        elif len(idx_nonempty) == 0:
            raise ValueError('all RefSets were empty')

    @property
    def n_overlaps(self):
        if self._n_overlaps is None:
            self._n_overlaps = self._overlap()
        return self._n_overlaps

    @property
    def pval(self):
        if self._pval_enrich is None:
            self._pval_enrich = self._test()
        return self._pval_enrich

    @property
    def qval(self, method="fdr_bh"):
        if self._qval_enrich is None:
            self._qval_enrich = self._correction(method=method)
        return self._qval_enrich

    def _overlap(self):
        """
        """
        self.RefSets.overlap(enq_set=self._significant_ids, set_id=None)
        return np.array([len(x.intersect) for x in self.RefSets.sets])

    def _test(self):
        """
        """
        pval = stats.hypergeom_test(
            intersections=self.n_overlaps,
            enquiry=len(self._significant_ids),
            references=self.RefSets._set_lens,
            background=len(self._all_ids)
        )
        return pval

    def _correction(self, method) -> np.ndarray:
        """
        Performs multiple testing corrections available in statsmodels.stats.multitest.multipletests()
        on self.pval.

        :param method: Multiple testing correction method.
            Browse available methods in the annotation of statsmodels.stats.multitest.multipletests().
        """
        return correction.correct(pvals=self.pval, method=method)

    ## Output functions.

    def grepv_sets(self, x):
        """ 
        Search gene set identifiers for a substring.
        """
        return self.RefSets.grepv_sets(x)

    def set(id):
        """ 
        Return the set with a given set identifier.
        """
        return self.RefSets.get_set(id)

    def significant_sets(self, threshold=0.05) -> list:
        """
        Return significant sets from gene set enrichement analysis as an output table.
        """
        return self.RefSets.subset(idx=np.where(self.qval <= threshold)[0])

    def significant_set_ids(self, threshold=0.05) -> np.array:
        """
        Return significant sets from gene set enrichement analysis as an output table.
        """
        return [self.RefSets._ids[i] for i in np.where(self.qval <= threshold)[0]]

    def summary(self) -> pd.DataFrame:
        """
        Summarize gene set enrichement analysis as an output table.
        """
        res = pd.DataFrame({
            "set": self.RefSets._ids,
            "pval": self.pval,
            "qval": self.qval,
            "intersection": self.n_overlaps,
            "reference": self.RefSets._set_lens,
            "enquiry": len(self._significant_ids),
            "background": len(self._all_ids)
        })
        # Sort by p-value
        res = res.iloc[np.argsort(res['pval'].values), :]
        return res
