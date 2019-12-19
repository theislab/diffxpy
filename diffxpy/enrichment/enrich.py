import logging
import numpy as np
import pandas as pd
import abc
import json
import os
import warnings
import urllib.request as request
from typing import Union, List, Dict

from ..stats import stats
from ..testing import correction
from ..testing.det import _DifferentialExpressionTest


class RefSets:
    """
    Class for a list of gene sets.

    Input:
    1. Read gene sets from file.
    2. Give a list with gene sets.
    3. Manually add gene sets one by one.

    .gmt files can be downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp for example.
    """

    class _Set:
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
            if len(sets) > 0:
                self.load_sets(sets, type=type)
                self._genes = np.sort(np.unique(np.concatenate([np.asarray(list(x.genes)) for x in self.sets])))
            else:
                self.sets = []
                self._genes = np.array([])
        elif fn is not None:
            self.read_from_file(fn=fn, type=type)
            self._genes = np.sort(np.unique(np.concatenate([np.asarray(list(x.genes)) for x in self.sets])))
        else:
            self.sets = []
            self._genes = np.array([])
        self._ids = np.array([x.id for x in self.sets])
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
        self._ids = np.array([x.id for x in self.sets])
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
        return self.sets[self._ids.tolist().index(id)]

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
            x.intersect = self.get_set(id).genes.intersection(enq_set)  # bug


class _GetInteractionsBase(abc.ABC):
    """
    This class defines an interface all GetInteraction class have to comply with
    """

    @abc.abstractmethod
    def get_interactions(self, ko_gene_list: List[str], only_physical_or_genetic_inters=True):
        """
        This method returns a dictionary containing the possible interactions for each element of the list on input
        gene names
        :param ko_gene_list: List[str]
            names of input genes e.g. ['FOXA2', 'OTX2', ...]
        :param only_physical_or_genetic_inters: bool
            if True -> only physical and genetic interactions are returned
            else -> all kind of interactions are returned
        :return: Dict[gene_name: str, List[interactions: str]]
            the elements of ko_gene_list are the keys of the return dictionary
            the entries for each key are the corresponding gene interactions
        """
        pass


class _GetInteractionsBioGrid(_GetInteractionsBase):

    def __init__(self, access_key: str):
        self.accessKey = access_key

    def __get_request(self, gene_name: str):
        url = f'https://webservice.thebiogrid.org/interactions?searchNames=true' \
            f'&geneList={gene_name}' \
            f'&includeInteractors=true' \
            f'&includeInteractorInteractions=false' \
            f'&selfInteractionsExcluded=true' \
            f'&taxId=9606' \
            f'&accesskey={self.accessKey}' \
            f'&format=json'

        return url

    def get_interactions(self, ko_gene_list: List[str], only_physical_or_genetic_inters=True):
        """
        See base class -> GetInteractionsBase
        """
        interactions = {}
        sources = {}

        for ko_gene in ko_gene_list:
            interaction_list = []
            sources[ko_gene.upper()] = {}
            # noinspection PyBroadException
            try:
                with request.urlopen(self.__get_request(gene_name=ko_gene.upper())) as response:
                    data = response.read()
                    data = json.loads(data)
                    for key in data.keys():
                        gene_a = str(data[key]['OFFICIAL_SYMBOL_A']).upper()
                        gene_b = str(data[key]['OFFICIAL_SYMBOL_B']).upper()
                        interaction_type = str(data[key]['EXPERIMENTAL_SYSTEM_TYPE']).lower()
                        # select the gene that is != ko_gene
                        if gene_a == ko_gene.upper():
                            gene = gene_b
                        else:
                            gene = gene_a

                        if only_physical_or_genetic_inters:
                            if interaction_type == 'physical' or interaction_type == 'genetic':
                                interaction_list.append(gene)
                                if gene not in sources[ko_gene.upper()]:
                                    sources[ko_gene.upper()][gene] = [{'PUBMED_AUTHOR': data[key]['PUBMED_AUTHOR'],
                                                                       'PUBMED_ID': data[key]['PUBMED_ID']}]
                                else:
                                    sources[ko_gene.upper()][gene] += [{'PUBMED_AUTHOR': data[key]['PUBMED_AUTHOR'],
                                                                        'PUBMED_ID': data[key]['PUBMED_ID']}]
                        else:
                            interaction_list.append(gene)
                            if gene not in sources[ko_gene.upper()]:
                                sources[ko_gene.upper()][gene] = [{'PUBMED_AUTHOR': data[key]['PUBMED_AUTHOR'],
                                                                   'PUBMED_ID': data[key]['PUBMED_ID']}]
                            else:
                                sources[ko_gene.upper()][gene] += [{'PUBMED_AUTHOR': data[key]['PUBMED_AUTHOR'],
                                                                    'PUBMED_ID': data[key]['PUBMED_ID']}]
            except Exception:
                warnings.warn(f'Literature database: Could not get any interactions for {ko_gene.upper()} gene')

            # remove redundant interactions
            interactions[ko_gene.upper()] = list(np.unique(interaction_list))

        return interactions, sources


class RefSetLoaderBioGrid:
    """
    This class loads a RefSet instance that is preloaded with interactions loaded from the BioGrid Web API

    Example code:

    access_key = 'your_biogrid_access_key'
    ko_name_list = ['FOXA2', 'ARNT', 'ZAP70']

    loader = RefSetLoaderBioGrid(access_key, ko_name_list)
    ref_sets = loader.get_ref_set()  # -> now you can use the ref_sets instance as you are used to

    # if you want to save the retrieved interactions and their sources
    loader.save_sources_to_file()
    loader.save_inters_to_file()
    """

    def __init__(self,
                 access_key: str,
                 ko_name_list: List[str],
                 only_physical_or_genetic_inters=True):
        """
        :param access_key: your BioGrid access key
            you can get one by filling out this form: https://webservice.thebiogrid.org/
        :param ko_name_list: List[str]
            name of the genes your knockouts are targeting
            e.g. : ['FOXA2', 'ARNT', 'ZAP70']
        :param only_physical_or_genetic_inters: bool
            True: only physical and genetic interactions will be considered
            False: all interactions will be considered
        """

        self.__inters, self.__sources = self.__get_biogrid_data(
            access_key=access_key,
            ko_name_list=ko_name_list,
            only_physical_or_genetic_inters=only_physical_or_genetic_inters
        )

        self.__ref_sets = self.__create_refset()

    @staticmethod
    def __get_biogrid_data(access_key: str,
                           ko_name_list: List[str],
                           only_physical_or_genetic_inters: bool):
        biogrid = _GetInteractionsBioGrid(access_key=access_key)
        inters, sources = biogrid.get_interactions(ko_gene_list=ko_name_list,
                                                   only_physical_or_genetic_inters=only_physical_or_genetic_inters)

        return inters, sources

    def __create_refset(self):
        ref_sets = RefSets()

        for key in self.__inters.keys():
            ref_sets.add(id=str(key), source='BioGrid', gene_ids=self.__inters[key])

        return ref_sets

    def get_refset(self):
        """
        This method returns the RefSets instance preloaded with the interactions retrieved from BioGrid
        :return: RefSets
        """
        return self.__ref_sets

    def save_sources_to_file(self, save_dir='', save_format='json'):
        """
        This method saves all the sources of the interactions to file
        :param save_dir: str
            Path to the directory where data should be saved to
        :param save_format: str
            either: 'json' --> better format as it preserves the structure of the data
            or: 'csv' --> flattens the data (not grouped by knockouts anymore)
        :return: None
        """
        if save_format == 'json':
            with open(os.path.join(save_dir, 'ref_data_sources.json'), 'w') as file:
                json.dump(self.__sources, file, indent=4, sort_keys=True)
        elif save_format == 'csv':
            save_list = [[], [], [], []]
            for key_ko_gene in self.__sources.keys():
                for key_inter in self.__sources[key_ko_gene].keys():
                    for source in self.__sources[key_ko_gene][key_inter]:
                        save_list[0] = save_list[0] + [str(key_ko_gene)]
                        save_list[1] = save_list[1] + [str(key_inter)]
                        save_list[2] = save_list[2] + [source['PUBMED_AUTHOR']]
                        save_list[3] = save_list[3] + [source['PUBMED_ID']]
            df = pd.DataFrame(np.array(save_list).T)
            df.to_csv(os.path.join(save_dir, 'ref_data_sources.csv'))
        else:
            raise ValueError('save_format you selected is not supported')
        raise NotImplementedError

    def save_inters_to_file(self, save_dir='', save_format='json'):
        """
        This method saves all the interactions retrieved from BioGrid to file
        :param save_dir: str
            Path to the directory where data should be saved to
        :param save_format: str
            either: 'json' --> better format as it preserves the structure of the data
            or: 'csv' --> flattens the data (not grouped by knockouts anymore)
        :return: None
        """
        if save_format == 'json':
            with open(os.path.join(save_dir, 'ref_data.json'), 'w') as file:
                json.dump(self.__inters, file, indent=4, sort_keys=True)
        elif save_format == 'csv':
            save_list = [[], []]
            for key in self.__inters.keys():
                inters = self.__inters[key]
                save_list[0] = save_list[0] + ([str(key)] * len(inters))
                save_list[1] = save_list[1] + inters
            df = pd.DataFrame(np.array(save_list).T)
            df.to_csv(os.path.join(save_dir, 'ref_data.csv'))
        else:
            raise ValueError('save_format you selected is not supported')


def test(
        ref: RefSets,
        det: Union[_DifferentialExpressionTest, None] = None,
        scores: Union[np.array, None] = None,
        gene_ids: Union[list, None] = None,
        threshold=0.05,
        incl_all_zero=False,
        all_ids=None,
        clean_ref=False,
        capital=True
):
    """ Perform gene set enrichment.

    Wrapper for Enrich. Just wrote this so that Enrich shows up with a
    nice doc string and that the call to this is de.enrich.test which
    makes more sense to me than de.enrich.Enrich.

    :param ref: The annotated gene sets against which enrichment is tested.
    :param det: The differential expression results object which is tested
        for enrichment in the gene sets.
    :param scores: Alternative to DETest, vector of scores (scalar per gene) which are then
        used to discretize gene list. This can for example be corrected p-values from a differential expression
        test, in that case the parameter threshold would be a significance threshold.
    :param gene_ids: If pval was supplied instead of DETest, use gene_ids to supply the
        vector of gene identifiers (strings) that correspond to the p-values
        which can be matched against the identifiers in the sets in RefSets.
    :param threshold: Threshold of parameter scores at which a gene is included as a hit: In the case
        of differential test p-values in scores, threshold is the significance threshold.
    :param incl_all_zero: Wehther to include genes in gene universe which were all zero.
    :param all_ids: Set of all gene identifiers, this is used as the background set in the
        hypergeometric test. Only supply this if not all genes were tested
        and are supplied above in DETest or gene_ids.
    :param clean_ref: Whether or not to only retain gene identifiers in RefSets that occur in
        the background set of identifiers supplied here through all_ids.
    :param capital: Make all gene IDs captial.
    """
    return Enrich(
        ref=ref,
        det=det,
        scores=scores,
        gene_ids=gene_ids,
        threshold=threshold,
        incl_all_zero=incl_all_zero,
        all_ids=all_ids,
        clean_ref=clean_ref,
        capital=capital
    )


class Enrich:
    """
    """

    def __init__(
            self,
            ref: RefSets,
            det: Union[_DifferentialExpressionTest, None],
            scores: Union[np.array, None],
            gene_ids: Union[list, np.ndarray, None],
            threshold,
            incl_all_zero,
            all_ids,
            clean_ref,
            capital
    ):
        self._n_overlaps = None
        self._pval_enrich = None
        self._qval_enrich = None
        if isinstance(gene_ids, list):
            gene_ids = np.asarray(gene_ids)
        # Load multiple-testing-corrected differential expression
        # p-values from differential expression output.
        if det is not None:
            if incl_all_zero:
                self._qval_de = det.qval
                self._gene_ids = det.gene_ids
            else:
                idx_not_all_zero = np.where(np.logical_not(det.summary()["zero_mean"].values))[0]
                self._qval_de = det.qval[idx_not_all_zero]
                self._gene_ids = det.gene_ids[idx_not_all_zero]
        elif scores is not None and gene_ids is not None:
            self._qval_de = np.asarray(scores)
            self._gene_ids = gene_ids
        else:
            raise ValueError('Supply either DETest or pval and gene_ids to Enrich().')
        # Take out NA genes labels:
        # Select significant genes based on user defined threshold.
        if any([x is np.nan for x in self._gene_ids]):
            idx_notnan = np.where([x is not np.nan for x in self._gene_ids])[0]
            logging.getLogger("diffxpy").info(
                " Discarded %i nan gene ids, leaving %i genes.",
                len(self._gene_ids) - len(idx_notnan),
                len(idx_notnan)
            )
            self._qval_de = self._qval_de[idx_notnan]
            self._gene_ids = self._gene_ids[idx_notnan]

        self._significant_de = self._qval_de <= threshold
        self._significant_ids = set(self._gene_ids[np.where(self._significant_de)[0]])
        if all_ids is not None:
            self._all_ids = set(all_ids)
        else:
            self._all_ids = set(self._gene_ids)

        if capital:
            self._gene_ids = [x.upper() for x in self._gene_ids]
            self._all_ids = set([x.upper() for x in self._all_ids])
            self._significant_ids = set([x.upper() for x in self._significant_ids])

        # Generate diagnostic statistic of number of possible overlaps in total.
        logging.getLogger("diffxpy").info(
            " %i overlaps found between refset (%i) and provided gene list (%i).",
            len(set(self._all_ids).intersection(set(ref._genes))),
            len(ref._genes),
            len(self._all_ids)
        )
        self.missing_genes = list(set(ref._genes).difference(set(self._all_ids)))
        # Clean reference set to only contains ids that were observed in
        # current study if required.
        self.RefSets = ref
        if clean_ref:
            self.RefSets.clean(self._all_ids)
        # Print if there are empty sets.
        idx_nonempty = np.where([len(x.genes) > 0 for x in self.RefSets.sets])[0]
        if len(self.RefSets.sets) - len(idx_nonempty) > 0:
            logging.getLogger("diffxpy").info(
                " Found %i empty sets, removing those.",
                len(self.RefSets.sets) - len(idx_nonempty)
            )
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

    def set(self, id):
        """ 
        Return the set with a given set identifier.
        """
        return self.RefSets.get_set(id)

    def significant_sets(self, threshold=0.05) -> list:
        """
        Return significant sets from gene set enrichement analysis as an output table.
        """
        sig_sets = np.where(self.qval <= threshold)[0]
        if len(sig_sets) == 0:
            logging.getLogger("diffxpy").info("no significant sets found")
        return self.RefSets.subset(idx=sig_sets)

    def significant_set_ids(self, threshold=0.05) -> np.array:
        """
        Return significant sets from gene set enrichement analysis as an output table.
        """
        return [self.RefSets._ids[i] for i in np.where(self.qval <= threshold)[0]]

    def summary(self, sort=True) -> pd.DataFrame:
        """
        Summarize gene set enrichement analysis as an output table.

        :param sort: Whether to sort table by p-value.
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
        if sort:
            res = res.iloc[np.argsort(res['pval'].values), :]
        return res

    def set_summary(self, id: str):
        """
        Summarize gene set enrichement analysis for a given set.
        :param id: Gene set to enquire.

        :return: Slice of summary table.
        """
        return self.summary(sort=False).iloc[self.RefSets._ids.tolist().index(id), :]
