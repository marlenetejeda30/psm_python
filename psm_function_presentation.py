#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:12:20 2024

@author: mtejeda
"""

import pandas as pd
import plotly.express as px
pd.options.plotting.backend = "plotly"
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

sup_style = """
    <style>
        .sup {
            font-size: medium;
        }
    </style>
"""

st.markdown(sup_style, unsafe_allow_html=True)

st.markdown('''
<style>
.katex-html {
    text-align: left;
}
</style>''',
unsafe_allow_html=True
)

with st.sidebar: 
	selected = option_menu(
		menu_title = 'Propensity Score Analysis (PSA)',
		options = ['Introduction','PSA Overview', 'Phase 1', 'Phase 2','Phase 3','Phase 4','References'],
		menu_icon = 'brightness-high-fill',
		icons = ['virus2', 'signpost-fill', 'bar-chart-line-fill', 'arrows-collapse-vertical', 'check-circle-fill','trophy-fill','bookmark-check-fill'],
		default_index = 0,
		)
    

if selected=="Introduction":
    st.title('Introduction')
    st.markdown('#### Recently propensity score analysis has been increasingly used to estimate causal effects in observational studies because it removes confounding factors. The propensity score is defined as the probability of treatment assignment conditional on observed baseline characteristics<sup class="sup">1</sup>.  Rubin defined the causal effect for an individual as the comparison between an individual’s outcome whether they received the treatment versus if they did not<sup class="sup">2</sup>. The propensity score allows one to design and analyze an observational (nonrandomized) study so that it mimics some of the characteristics of a randomized controlled trial<sup class="sup">1</sup>. To solve the cause and result relationship between HSV-1 and risk of AD, we performed PSA. To conduct PSA, we developed an algorithm that includes the four phases that Stuart postulated<sup class="sup">3</sup>',unsafe_allow_html=True)


if selected=='PSA Overview':
    st.title("Propensity Score Analysis (PSA) Overview")
    
    
    st.markdown('#### To conduct PSA, we developed an algorithm with four phases based on those that Stuart posited<sup class="sup">3</sup>.  The first phase includes selecting the pre-treatment and assessing overlap between groups by relying on several pre-match metrics and statistics. The second phase produces several different matched sets using different methods. Phase 3 assesses the quality of the matched sets based on how well covariates balance while having sufficient treated individuals in the matched set. If none of the sets in phase 3 are satisfactory then either more matching algorithms are required or a different set of covariates and/or interactions and squared terms should be used. Note that the matched set should not be chosen based on the outcome variable (phases 1-3 are iterative).  In the fourth phase, the best matched set is then used to preform outcome analysis and estimation of the treatment effect. The remainder of this section describes each of these four phases in greater detail.',unsafe_allow_html=True)
    
    st.markdown("#### The algorithm is implemented in a python class object and in order to instatiate the class you must provide a dataframe that contains all of the potential matching covariates along with the binary treatment variable and the outcome variable. In addition a file path pointing to the root directory is necessary for creating the folder structure for saving subsequent results. Below is an example of the code required to instatiate the class: ")
    
    st.code('''cov_keep = ["AD","HSV1","PC_Race_Label","BODY_SITE","SEQ_CENTER","APOE","PCR_free","Sex","Age", "Sequencer", "Sequencing_Platform"]
my_psm = PSA(main_df[cov_keep],treat_var= "HSV1",target_var= "AD",folder_path= "path/to/root/folder")''')
    
    st.markdown("#### If you are interested to see the how the init method of the PSA class works, click below to see the following methods associated with this initial command:")
    with st.expander("##### Main Init Method"):
        
        
        init_code = '''        class PSA:
            def __init__(self, df=None,treat_var=None, folder_path=None,target_var=None, impute_method = "Rosenbaum",prevent_reverse=False,  manager=None):
                if df is None or treat_var is None or folder_path is None:
                    print('Nothing added so either add arguments for "df", "target_var", and "folder_path" by another instantiation or use the load_old_results to reload old model!')
                    self.name = "myPSM"
                    return None
                
                self.main_path = self._process_main_file_path(folder_path) # function creates pickle path too
                self.logger = self._setup_logger(init=True)
                
                
                assert isinstance(df,pd.DataFrame)
                #self.manager = Manager()
                ##Impute missings if necessary
                
                self.time_stamp = self._generate_time_stamp()

                treat_var = treat_var.replace(".","_")
                target_var = target_var.replace(".","_")
                df.columns = df.columns.str.replace(".","_",regex=True)
                
                #Identify categorical and numerical variables
                #self.logger.info("Here are the columns: ",df.columns)
                
                self.cat_vars = list(df.drop([target_var,treat_var],axis=1).select_dtypes(include = ['object','category']).columns)
                
                
                self.num_vars = [col for col in df.drop([target_var,treat_var],axis=1).columns if col not in self.cat_vars]
                
                self.masterDf = self._impute_missing(df.copy(), method = impute_method)
                
                #Convert the treat variable to control matching behavior
                self.treat_var = treat_var
                self.reverse_groups = False
                self.treat_keys = None
                self.masterDf[treat_var] = df[treat_var]
                self.masterDf[treat_var] = self._process_treat_var(treat_var,prevent_reverse)
                
                ##Process target variable
                self.target_enc = sklearn.preprocessing.LabelEncoder()
                self.outcome_analysis_valid = True
                ##Create dummy target var but forbid outcome analysis at later stages
                if target_var is None:
                    self.logger.warning('No target variable indicated means no outcome analysis can be performed after matching!')
                    #Creates equal distribution of dummy categories and samples half of them
                    self.masterDf['dummy_target'] = random.sample([0,1]*len(df),len(df))
                    target_var = 'dummy_target'
                    self.outcome_analysis_valid = False
               
                self.masterDf[target_var] = df[target_var]
                self.target_var = self._process_target_var(target_var)
                
                #Create indicator variables for cat vars also makes self.indicator_vars for convenience
                self.masterDf = self._process_cat_vars()
                
                #Create match_id variable necessary for creating the matches
                self.match_id = "match_id"
                self.masterDf = self._create_match_id_col(self.masterDf)
                
                
                self.first_order_pval = stats.norm.sf(1) 
                self.second_order_pval = 0.05
                self.base_preds = []
                self.ps_estimators = {f'thresh {self.first_order_pval:.4f},0.05':{}}
                self.best_model_meta = {f'thresh {self.first_order_pval:.4f},0.05':[]}
                #self.psm_df = {f'thresh {self.first_order_pval:.4f},0.05':[]}
                self.desc_stats = {}
                self.round_digits = 4
                self.name = self.file_prefix
                self.random_seed = 33
                #self.n_jobs= cpu_count()-5
                self.n_jobs = 8
                if manager is None:
                    self.matched_sets={}
                else:
                    self.matched_sets = manager.dict()
                    self.lock = manager.Lock()
                #self.save_results()
                self.logger.info(f'Currently the following variables will be treated as categorical: {list(self.cat_vars)}  If there are mistakes please convert to "object" or "category" prior to instantiation.')
            
            def __repr__(self):
               return f'PSA_1.3("{self.name}" at "{self.main_path}")'
           
            def __str__(self):
                return f'PSA_1.3 ({self.name} at {self.main_path})' '''
        
        st.code(init_code,language="python")
            
    with st.expander("##### Helper 1: process_main_file_path"):
        
        st.markdown('''*This checks to make sure the parent folder input is correctly specified and has no file extensions. 
        It must be a directory. Extremely important because this acts as the file naming and storing root directory. 
        Creates an attribute called self.file_prefix which will be used to make file names unique.* ''')
        process_main_file_path = """        def _process_main_file_path(self,folder_path):
    if r"/" in folder_path:
        #If a file path is entered
        full_path= folder_path.split("/")
        pkl_file_name = full_path.pop()
        file_ind = pkl_file_name.find(r'.')
        #Special case: if user supplies a file path that has a file extension
        if file_ind!=-1:
            #Converts file name to directory with same name stripped of extension
            last_folder = pkl_file_name[0:file_ind]
            folder_path = "/".join(full_path)+'/'+last_folder
            print('Invalid folder path. Did you enter a file instead of folder?')
            if Path(folder_path).exists():
                raise ValueError(f'Folder already exists! If you wish to overwrite, enter the following for output_file_path:\n {folder_path}')
            
            print(f'Creating the following folder instead:\n{folder_path}')
        #Normal Case
        else:
            last_folder=pkl_file_name
    
    #User enters folder only - prepends the current directory to folder
    else:
        file_ind = folder_path.find(r'.')
        if file_ind!=-1:
            #Converts file name to directory with same name stripped of extension
            print('Invalid folder path. Did you enter a file instead of folder?')
            last_folder = folder_path[0:file_ind]
            folder_path = str(Path().cwd()) + "/" + last_folder
            if Path(folder_path).exists():
                raise ValueError(f'Folder already exists! If you wish to overwrite, enter the following for output_file_path:\n {folder_path}')
        #Normal Case
        else:
            last_folder = folder_path
            folder_path = str(Path().cwd()) + "/" + last_folder
        print(f'No path given: using the following current directory to store all results: {folder_path}')
    
    ###Creates file path if none exists
    if not Path(folder_path).is_dir():
        Path(folder_path).mkdir(mode=0o700,parents=True,exist_ok=False)
        print(f'Creating the following folder path to store all results: {folder_path}')
    
    self.file_prefix = last_folder
    
    return(folder_path) """
        
        st.code(process_main_file_path,language="python")
        
        
    with st.expander("##### Helper 2: setup_logger"):
        setup_logger = """def _setup_logger(self, init=False, new_directory=None):
    #check if file path has changed
    if new_directory is not None:
        self.main_path = new_directory
    ###Creates file path if none exists
    folder_path = self.main_path + '/log'
    if init or new_directory is not None:
        if not Path(folder_path).is_dir():
            Path(folder_path).mkdir(mode=0o700,parents=True,exist_ok=False)
        
    logger = get_logger()
    logger.setLevel(logging.INFO)
    # create a file handler to write to a log file
    # Get the current process name and PID
    process_name = current_process().name
    process_pid = current_process().pid
    
    log_file = f"{folder_path}/process_{process_name}_{process_pid}.log"

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # create a stream handler to only write errors to the console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.ERROR)
    
    #Create formatter for both
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n\n')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    #Add warnings filter for file handler
    dup_filter = DuplicateFilter()
    fh.addFilter(dup_filter,)
    ch.addFilter(dup_filter)
    logging.captureWarnings(True)

    ##Add both handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger """

        st.code(setup_logger,language="python")
    with st.expander("##### Helper 3: impute_missing"):

        impute_missing = """def _impute_missing(self,df,method = "Rosenbaum"):
#Check if any missing and if not return dataframe unchanged
if all(df.isnull().sum()==0):
    return df

#Handle categorical vars
if not all(df[self.cat_vars].isnull().sum()==0):
    cat_imputer = SimpleImputer(strategy="constant",fill_value="missing")
    if len(self.cat_vars)==1:
        
        cat_df = pd.DataFrame(cat_imputer.fit_transform(df[self.cat_vars].assign(ind="hi")),columns = cat_imputer.feature_names_in_)
        cat_df.drop("ind",axis=1,inplace=True)
    else:
    
        cat_df = pd.DataFrame(cat_imputer.fit_transform(df[self.cat_vars]),columns = cat_imputer.feature_names_in_)
else:
    cat_df = df[self.cat_vars].copy()

#Handle numeric vars
if not all(df[self.num_vars].isnull().sum()==0):
    if method == "Rosenbaum":
        num_df = df[self.num_vars].copy()
        for col in num_df.columns:
            if num_df[col].isnull().sum()>0:
                #Create indicator column for whether col is missing
                num_df[str(col)+"_missing"] = np.where(num_df[col].isnull(),1,0)
                self.indicator_vars+=[str(col)+"_missing"]
                #Add 5 to the max value of current variable as per Rosenbaum 2010
                max_col_val  = np.ceil(num_df[col].max())+5
                num_df[col] = num_df[col].replace({np.nan: max_col_val})
else:
    num_df = df[self.num_vars].copy()

return pd.concat([num_df,cat_df],axis=1) """
        st.code(impute_missing,language="python")


    with st.expander("##### Helper 4: process_treat_var"):
        st.markdown("*This function checks that the treat variable has some variability. Also encodes as 0-1 variable which is neccesary for later results.*")
        
        process_treat_var = """def _process_treat_var(self,treat_var,prevent_reverse):
    assert isinstance(treat_var,str)
    if len(self.masterDf[treat_var].unique())<2:
        self.logger.exception(f'{treat_var} has one unique group so matching is not possible')
        raise ValueError
    #Checks if only 1 case which will cause error in train splits
    if self.masterDf[treat_var].value_counts().min()<2:
        self.logger.exception(f'{treat_var} has less than two cases in one group.')
        raise ValueError
    
    #Force string conversion to eliminate floats
    processed_treat_var = self.masterDf[treat_var].astype(str)
    treat_var_counts = collections.Counter(processed_treat_var)
    #Guarantee the order of unique
    unique_vals = sorted(processed_treat_var.unique())
    #Set default of reverse_groups to False - only change to true if necessary
    self.reverse_groups = False
    
    if prevent_reverse:
        #No need to reorder so use label encoder
        treat_enc = sklearn.preprocessing.LabelEncoder()
        processed_treat_var = treat_enc.fit_transform(self.masterDf[treat_var])
        self.treat_keys = {num: key for num, key in enumerate(treat_enc.classes_)}
        ##Extract the string that will be considered the treatment group
        treatment_key = self.treat_keys[1]
        self.logger.warning(f'The group indicated by the higher number {self.treat_keys[0]} has more observations but you indicated not to change. This will limit the types of matching available!')
    else:
        ###Check if control > treatment group
        if treat_var_counts[unique_vals[1]]>treat_var_counts[unique_vals[0]]:
            treat_enc = sklearn.preprocessing.OrdinalEncoder(categories=[[unique_vals[1],unique_vals[0]]])
            if is_numeric_dtype(self.masterDf[treat_var]):
                processed_treat_var = pd.Series(treat_enc.fit_transform(processed_treat_var.to_numpy().reshape(-1,1)).flatten()).astype(int)
                
                #Create dictionary with label keys - the categories attribute is a list of ndarray
                self.treat_keys = {num: key for num, key in enumerate(treat_enc.categories_[0])}
                treatment_key = self.treat_keys[1]
                #If numeric and higher number greater than lower number (i.e., 1>0) must reverse treatment Control
                self.reverse_groups = True
                self.logger.warning(f'The group indicated by the higher number {self.treat_keys[0]} has more observations and so will be treated as the comparison group in order for the matching algorithm to work.')
            else:
                processed_treat_var = pd.Series(treat_enc.fit_transform(processed_treat_var.to_numpy().reshape(-1,1)).flatten()).astype(int)
                
                #Create dictionary with label keys - the categories attribute is a list of ndarray
                self.treat_keys = {num: key for num, key in enumerate(treat_enc.categories_[0])}
                treatment_key = self.treat_keys[1]
                
        #No need to reorder so use label encoder
        else:
            treat_enc = sklearn.preprocessing.LabelEncoder()
            processed_treat_var = treat_enc.fit_transform(self.masterDf[treat_var])
            self.treat_keys = {num: key for num, key in enumerate(treat_enc.classes_)}
            ##Extract the string that will be considered the treatment group
            treatment_key = self.treat_keys[1]
            
    ##Check if the original values were 0, 1
    if set(self.treat_keys.values())==set(('0','1')):
        for k,v in self.treat_keys.items():
            if v=='0':
                self.treat_keys.update({k: (v, "Comparison")})
            else:
                self.treat_keys.update({k: (v, "Treatment")})
    else:
        for i, (k,v) in enumerate(self.treat_keys.items()):
            if i==0:
                self.treat_keys.update({k:(v, "Comparison")})
            else:
                self.treat_keys.update({k:(v, "Treatment")})
    
    if not is_numeric_dtype(self.masterDf[treat_var]):
        self.logger.warning(f'Treatment variable {treat_var} detected as string and will be converted to numerical with the following replacements: {self.treat_keys} This is because the category with the fewest values, {treatment_key} must be the treated group for the matching to work properly.')
    
    return(processed_treat_var) """

        st.code(process_treat_var,language="python")
    
    with st.expander("##### Helper 5: process_target_var"):
        process_target_var = """def _process_target_var(self,target_var):
    assert isinstance(target_var,str)
    #No variance case
    if self.masterDf[target_var].nunique()<2:
        self.logger.exception(f'{target_var} has one constant value so no variation to model')
        raise ValueError
    ##Numeric case
    elif self.masterDf[target_var].nunique()>2:
        self.target_var_type = 'numeric'
    ##Categorical Case
    else:
        self.target_var_type = 'binary'
        
        #Checks if only 1 case which will cause error in train splits
        if self.masterDf[target_var].value_counts().min()<2:
            self.logger.exception(f'{target_var} has less than two cases in one group. Cannot split the data into train and test splits')
            raise ValueError
        if not is_numeric_dtype(self.masterDf[target_var]):
            self.masterDf[target_var] = self.target_enc.fit_transform(self.masterDf[target_var])
            target_keys = {key: num for num, key in enumerate(self.target_enc.classes_)}
            self.logger.warning(f'Target variable {target_var} detected as string and will be converted to numerical with the following replacements: {target_keys} If this is not what you want recode your dataframe prior to using this ML classifier class.')
    
    return(target_var) """
        st.code(process_target_var,language="python")
        
    with st.expander("##### Helper 6: process_cat_vars"):
        process_cat_vars = """def _process_cat_vars(self):
    if len(self.cat_vars)>0:
        cat_df = pd.get_dummies(self.masterDf[self.cat_vars])
        ##Need to remove any periods in column names
        cat_df.columns = cat_df.columns.str.replace(".","_",regex=True)
        self.indicator_vars = list(cat_df.columns)
        ##Extract the name of first indicator var for each cat var to include in ref_vars
        ref_vars = []
        for cat in self.cat_vars:
            for indvar in self.indicator_vars:
                if str(cat)+"_" in indvar:
                    ref_vars.append(indvar)
                    break
        self.ref_vars = ref_vars
        return(pd.concat([self.masterDf,cat_df],axis=1))
    else:
        self.indicator_vars = []
        self.ref_vars = []
        return(self.masterDf) """
    
        st.code(process_cat_vars,language="python")
        
        
    with st.expander("##### Helper 7: create_match_id_col"):
        create_match_id = """   def _create_match_id_col(self,main_df):
        #Check if index is unique list of consecutive row_numbers that are needed for matches
        if not isinstance(main_df.index, pd.core.indexes.range.RangeIndex):
            #Index is not unique so need to reset_index and drop current one
            if len(main_df)!=main_df.index.nunique():
                #Check if dataframe has numeric index but may not be consecutive if dataframe was subsetted
                #if isinstance(main_df.index, pd.core.indexes.numeric.Int64Index):
                if main_df.index.dtype=="int64":
                    self.logger.warning('Current index is not unique which will lead to problems with matching functions! Replacing current index with a range of consecutive integers. If surprised you may wish to examine the dataframe before proceeding to run analysis to make sure dataframe is not corrupted.')
            ##If index is something else like character vector
                else:
                    self.logger.warning('Current index is not numeric or unique which will lead to problems with matching functions! Replacing current index with a range of consecutive integers. If surprised you may wish to examine the dataframe before proceeding to run analysis to make sure dataframe is not corrupted.')
                ##Replace index
                main_df.reset_index(drop=True,inplace=True)
        #Make sure no column is named match_index (unlikely) but will append number to end until no match
        if self.match_id in main_df.columns:
            match_id_replace = 'match_id'
            i = 1
            while True:
                match_id_replace = match_id_replace + str(i)
                if match_id_replace not in main_df.columns:
                    break
                i+=1
            
            self.logger.warning(f'You have a column named "match_id" in dataframe so changing the match_id to {match_id_replace} for creating matched sets')
            self.match_id = match_id_replace
        
        #Create index column named the value in self.match_id
        main_df.index.names = [self.match_id]
        return main_df.reset_index() """
        
        st.code(create_match_id,language="python")
        
        
        
        
        
        
        
        
        
        
        
if selected=='Phase 1':
    st.title("Phase 1: Selecting Covariates and Establishing Baseline Metrics")
    
    st.header('Selecting the Covariates')
    
    st.markdown('#### Variables that were affected by the treatment were not included and instead we chose variables that were associated with the treatment and/or outcome. Appropriateness of the score was assessed by examining the degree of similarity between the treated individuals and the controls after matching<sup class="sup">4</sup>.',unsafe_allow_html=True)
    
    st.markdown('#### To select these pre-treatment variables, we tested several models by implementing a stepwise variable selection procedure including interactions and quadratic terms described by Imbens and Rubin<sup class="sup">5</sup>.  Models varied depending on the essential covariates identified for estimating the propensity score. We used Imbens and Rubin’s automated procedure to select other variables that were deemed important in estimating the propensity score<sup class="sup">5</sup>. This includes adding an additional variable to the model to test whether it produces a large likelihood ratio test statistic compared to the other variables being tested and examine if it has a chi-square statistic greater than 1. This more liberal threshold is justified because the bias created by excluding important confounders is more costly than the slight increase in variance caused by including a variable unassociated with the outcome<sup class="sup">3</sup>.  The same automated procedure was used to test the inclusion of second-order terms (e.g., quadratic and interaction terms) of the chosen linear covariates. The only difference is that the authors recommend a more stringent test for second-order terms where the chi-square statistic must exceed 2.71, corresponding to a z-score of 1.645.',unsafe_allow_html=True)
    
    
    st.markdown('#### The following lines of code will select the covariates using the algorithm in my PSA class: ')
    
    st.code("""#Search for everything no restrictions
my_psm.determine_ps_feats() 
#Must include all categories of Race, Body, and APOE, Age but test for other covariates
my_psm.determine_ps_feats(keep=['PC_Race_Label','BODY_SITE','APOE','Age'])""")

    st.markdown("#### If you are interested to see the how the determine_ps_feats method of the PSA class works, click below to view it:")
    
    with st.expander('##### Main Function: determine_ps_feats'):
        st.markdown('*This will find the subset of features for estimating the PS as described in Imbens & Rubin (2015). It will also save the PS for each stepwise model along with the LL value. The attribute self.ps_estimators is dictionary with keys based on significance threshold and then a dictionary with keys as tuples for predictors included*')
        
        determine_ps_feats = """def determine_ps_feats(self,keep=None,exclude = None,first_order_pval = stats.norm.sf(1), second_order_pval = 0.05,allow_refs = False, all_interactions = True, all_squared=True,include_interactions = None, include_squared = None):
            
    self.first_order_pval = first_order_pval
    self.second_order_pval = second_order_pval
    ##Create pval_key based on user entered criteria for decision rules. If value is non-terminating round
    pval_key = self._convert_pval_key(first_order_pval, second_order_pval)
    
    ##Check if key for threshold already exists in dictionary self.ps_estimators and if not create one
    if pval_key not in self.ps_estimators:
        self.ps_estimators[pval_key] = {}
    #Add keep vars to self.base_preds list and display warning if they already exist.
    base_ref = []
    if keep is not None:
        assert isinstance(keep, (list,tuple,np.ndarray))
        if len(self.base_preds)>0:
            self.base_preds = []
        for var in keep:
            ###Need to make sure ref var is removed from cat_vars
            if var in self.cat_vars:
                cat_inds = []
                for ind in self.indicator_vars:
                    if ind.startswith(var):
                        if ind not in self.ref_vars:
                            cat_inds.append(ind)
                        else:
                            base_ref.append(ind)
                self.base_preds = self.base_preds + cat_inds
            else:
                self.base_preds.append(var)
    ##Get first order variables            
    first_order, current_ll = self._process_first_order(pval_key,exclude=exclude, first_order_pval = first_order_pval)
   
    #If model produced NaN!
    if np.isnan(current_ll):
        self.base_preds=[]
        return None
    ###Now get second order after first order
    if not second_order_pval:
        ps_df = self.masterDf[[self.match_id,self.target_var,self.treat_var]+first_order]
        final_vars = first_order
    else:
        ps_df, final_vars = self._process_second_order(first_order, current_ll, pval_key, second_order_pval, all_interactions, all_squared,include_interactions, include_squared)
   
    ps_df = pd.concat([ps_df,self.masterDf[base_ref]],axis=1)
    
    if pval_key not in self.best_model_meta:
        
        self.best_model_meta[pval_key] = []

    #Check for new second order columns to add to master df    
    new_columns = np.setdiff1d(ps_df.columns, self.masterDf.columns)
    if len(new_columns)>0:
        
        self.masterDf[new_columns] = ps_df[new_columns]
    
    ##For the df, need to maintain the order of the variables in the model keys embedded in the class
    sorted_predvars = sorted(final_vars)
    var_order = [self.match_id,self.target_var,self.treat_var] + sorted_predvars
    
    ps_df = ps_df[var_order]
    ps_df['psm_score'] = self.ps_estimators[pval_key][str(sorted_predvars)][1]
    
    model_info = 'during creation of the linearized propensity score for automatic feature selection'
    ps_df['lpsm_score'] = self._calc_linear_psm(str(sorted_predvars),model_info,self.ps_estimators[pval_key][str(sorted_predvars)][1])
    
    #Existing best_model_meta keys
    all_best_model_keys = list(map(lambda x: x[0],self.best_model_meta[pval_key]))

    #Checks for duplicates and if not will produce ValueError
    try:
        dup_idx = all_best_model_keys.index(str(sorted_predvars))
        self.logger.warning(f'The current model produced by your criteria already exists in the key "{pval_key}" at index {dup_idx}')
    
    #Value Error is actually a good thing because it means the model produced was not duplicated
    except ValueError:
        
        ##Determine which variables (especially second-orders) are binary or continuous
        #Get column number
        num_indexes = list(set(np.concatenate([np.where(ps_df.columns.str.contains(x))[0] for x in self.num_vars])))
        
        #Create boolean mask
        is_binary = np.array([False if i in num_indexes else True for i,_ in enumerate(ps_df.columns)])
        
        #Add the psm_score and lpsm_score to non binary variables (intentionally always the last two in df)
        is_binary[-2:] = False
        
        self.best_model_meta[pval_key].append((str(sorted(final_vars)),is_binary))
        
        ########Create Model with all indicators but ref var
        initial_best_model_key = str(sorted(final_vars))
        if len(self.cat_vars)>0:
            all_ind_dict = self._get_all_indicators(sorted_predvars,is_binary=is_binary, allow_refs=False)
            new_clean_vars = list(all_ind_dict.keys())
            sorted_col_ind = np.argsort(new_clean_vars)
            new_clean_vars = list(np.take(np.array(new_clean_vars),sorted_col_ind))
            
            #Check to see if already in best_model_meta
            all_best_model_keys = list(map(lambda x: x[0],self.best_model_meta[pval_key]))
            if str(new_clean_vars) not in all_best_model_keys:
                is_binary = list(np.take(np.array(list(all_ind_dict.values())),sorted_col_ind))
                is_binary = [True]*3  + is_binary + [False]*2
                
                self.best_model_meta[pval_key].append((str(new_clean_vars),np.array(is_binary)))
                ##Next check to see if this model already exists in ps_estimators if so do nothing but if not then must estimate the ps!
                all_model_keys = self.ps_estimators[pval_key].keys()
                if str(new_clean_vars) not in all_model_keys:
                    #define predictor variables
                    x1 = self.masterDf[new_clean_vars]
                    #add constant to predictor variables
                    x1 = sm.add_constant(x1)
                    #fit regression model
                    try:
                        model = sm.GLM(self.masterDf[self.treat_var], x1,family = sm.families.Binomial()).fit(disp=False)
                        self.ps_estimators[pval_key].update({str(new_clean_vars): (model.llf,model.predict())})
                    except RuntimeWarning as e:
                        bad_model_key = self.best_model_meta[pval_key].pop(-1)
                        self.logger.warning(f'The all indicator model with no reference variables for initial model {bad_model_key[0]} resulted in complete separation and the propensity score could not be estimated due to the following error: {e}. Model cannot be used for matching.')
                        
    finally:
        ##Need to reset base_preds or otherwise next model will contain old base_preds!!            
        self.base_preds = []
        
    return(ps_df)
 """

        st.code(determine_ps_feats,language="python")
    
    
    with st.expander('##### Helper 1: process_first_order'):
        
        process_first_order = """def _process_first_order(self,pval_key,exclude=None, first_order_pval = stats.norm.sf(1)):
    ##Run and store results of base_model
    current_base_vars = self.base_preds
    all_model_keys = self.ps_estimators[pval_key].keys()
    #Checks if base key was run before
    if str(sorted(self.base_preds)) not in all_model_keys:
        base_model = self._run_compare_glm(self.base_preds,self.masterDf,pval_key)
        if np.isnan(base_model[1]):
            self.logger.warning(f'The following model produced complete separation when estimating propensity scores for {self.treat_var}: {base_model[0]}   Try dropping a covariate and re-running!')
            return literal_eval(base_model[0]), base_model[1]
        
        base_model_key = base_model[0]
        ##Checks the case if user enters no other keep variables so the only one is constant which is returned as string and not list
        current_best_vars = literal_eval(base_model_key)
        
        #Uploads model to ps_estimators
        self.ps_estimators[pval_key].update({str(sorted(literal_eval(base_model_key))): (base_model[1],base_model[2])})
        base_ll = base_model[1]
    else:
        base_model = (str(sorted(self.base_preds)),*self.ps_estimators[pval_key][str(current_base_vars)])
        base_ll = base_model[1]
        current_best_vars = current_base_vars
    
    ##Exclude vars from being tested in model
    exclude = []
    if exclude is not None:
        assert isinstance(exclude, (list,tuple,np.ndarray))
        for var in exclude:
           ###Need to make sure ref var is removed from cat_vars
           if var in self.cat_vars:
               cat_inds = [ind for ind in self.indicator_vars if ind.startswith(var) and ind not in self.ref_vars]
               exclude = exclude + cat_inds
           else:
               exclude.append(var)
    
    #Remove the treatment variable, outcome variable, cat_vars that are not dummies and excluded vars
    existing_interactions = list(self.masterDf.columns[self.masterDf.columns.str.contains("[.]",regex=True)])
   
    test_vars = list(np.setdiff1d(self.masterDf.columns,[self.treat_var,self.target_var,self.match_id] + self.cat_vars + exclude + self.base_preds + existing_interactions))
    
    tmp_df = self.masterDf[self.base_preds + test_vars].copy()
    current_ll = base_ll
    tmp_best_model = base_model
    while len(test_vars)>0:
        #Checks if first model is constant only which will not be in the dataframe index
        if current_best_vars==["constant"]:
            model_var_list = [[i] for i in test_vars]
        else:
            model_var_list = [current_best_vars + [i] for i in test_vars]
        
        ##Return a list of tuples with 4 elements (model name, ll stat, ps_estimators, pval from test)
        func_kwargs = {'df':tmp_df,'pval_key':pval_key, "base_lr": current_ll}
       
        func = partial(self._run_compare_glm, **func_kwargs)
        with Pool(processes=self.n_jobs) as pool:
            all_glm_results= pool.map(func, model_var_list)

        all_lr_pvals = [pval[3] for pval in all_glm_results]
        if all(np.isnan(all_lr_pvals)):
            ##Check for duplicates and if so return np.nan
            
            all_model_keys = self.ps_estimators[pval_key].keys()
            tmp_model_key = str(sorted(literal_eval(tmp_best_model[0])))
            if tmp_model_key not in all_model_keys:
                self.ps_estimators[pval_key].update({tmp_model_key: (tmp_best_model[1],tmp_best_model[2])})
            break
        tmp_best_model = all_glm_results[np.nanargmin([pval[3] for pval in all_glm_results])]
        if tmp_best_model[3]<first_order_pval:
            all_model_keys = self.ps_estimators[pval_key].keys()
            tmp_model_key = str(sorted(literal_eval(tmp_best_model[0])))
            if tmp_model_key not in all_model_keys:
                self.ps_estimators[pval_key].update({tmp_model_key: (tmp_best_model[1],tmp_best_model[2])})
            current_ll = tmp_best_model[1]
            current_best_vars = literal_eval(tmp_best_model[0])
            test_vars.remove(current_best_vars[-1])
        else:
            break
    return(current_best_vars,current_ll) """

        st.code(process_first_order,language="python")
        
        
    with st.expander('##### Helper 2: run_compare_glm'):
    
        st.markdown('''*This function runs a glm and returns the variables used as a string, log-likelihood, entire model*''')
        run_compare_glm = """ def _run_compare_glm(self,pred_list,df,pval_key, base_lr=None):
    ###Checks if intercept only which is case if user enters no base variables
    if len(pred_list)>0:
        #define predictor variables
        x1 = df[pred_list]
        #add constant to predictor variables
        x1 = sm.add_constant(x1)
    else:
        x1 = np.ones(len(df))
        pred_list = ['constant']
    ##Check if model already run
    if str(pred_list) in self.ps_estimators[pval_key]:
        current_est = self.ps_estimators[pval_key][str(pred_list)]
        
        ##This means no model comparison being made so it will be the first model tested and nothing to compare to but model had already been run
        if base_lr is None:
            return(str(pred_list),current_est[0],current_est[1])
        #Compare new to old
        else:
            LR_stat = -2*(base_lr-current_est[0])
            p_val = stats.chi2.sf(LR_stat, 1)
            return(str(pred_list),current_est[0],current_est[1],p_val)
    #fit regression model
    final_results= None
    try:
        model = sm.GLM(self.masterDf[self.treat_var], x1,family = sm.families.Binomial()).fit(disp=False)
        final_results = [str(pred_list), model.llf,model.predict()]
    except RuntimeWarning:
        pass
    finally:
        if final_results is None:
            final_results = [str(pred_list),np.nan, np.repeat(np.nan, x1.shape[0])]
        
        if base_lr is None:
            return final_results
        if np.isnan(final_results[1]):
            return final_results + [np.nan]
        #Do comparison if base_lr provided
        LR_stat = -2*(base_lr-model.llf)
        p_val = stats.chi2.sf(LR_stat, 1)
        return final_results + [p_val]"""
    
        st.code(run_compare_glm,language="python")
    
    
    with st.expander('##### Helper 3: process_second_order'):
        process_second_order = """def _process_second_order(self, first_order, current_ll, pval_key, second_order_pval = 0.05, all_interactions = True, all_squared=True,include_interactions = None, include_squared = None):
    interact_df = self._generate_interactions(first_order, all_interactions, all_squared,include_interactions,include_squared)
    #Create list of test vars based on the remaining columns
    test_vars = list(interact_df.columns)
    #Create dataframe used for model selection
    tmp_df = pd.concat([self.masterDf,interact_df],axis=1)
    #Remove potential duplicate interactions created from previous runs
    tmp_df = tmp_df.iloc[:,~tmp_df.columns.duplicated()]
    #Initialize current_best_vars to first_order
    current_best_vars = first_order
    while len(test_vars)>0:
        model_var_list = [current_best_vars + [i] for i in test_vars]
        func_kwargs = {'df':tmp_df,'pval_key':pval_key, "base_lr": current_ll}
        func = partial(self._run_compare_glm, **func_kwargs)
        #Run all glms with additional test var in parallel
        with Pool(processes=self.n_jobs) as pool:
            all_glm_results= pool.map(func, model_var_list)
        p_val = [stats.chi2.sf(-2*(current_ll - llf[1]),1) for llf in all_glm_results]
        #Check if all are np.nan
        if all(np.isnan(p_val)):
            ##Check for duplicates and if so return np.nan
            all_model_keys = self.ps_estimators[pval_key].keys()
            tmp_model_key = str(sorted(current_best_vars))
            if tmp_model_key not in all_model_keys:
                self.logger.warning('Something went horribly wrong with first order results and they werent stored correctly!')
            break
       
        else:
            best_index = np.nanargmin(p_val)
            tmp_best_model = all_glm_results[best_index]
            
            if p_val[best_index]<second_order_pval:
                ##Check for duplicates and if so return np.nan
                all_model_keys = self.ps_estimators[pval_key].keys()
                tmp_model_key = str(sorted(literal_eval(tmp_best_model[0])))
                if tmp_model_key not in all_model_keys:
                    self.ps_estimators[pval_key].update({tmp_model_key: (tmp_best_model[1],tmp_best_model[2])})
                current_ll = tmp_best_model[1]
                current_best_vars = literal_eval(tmp_best_model[0])
                test_vars.remove(current_best_vars[-1])
            else:
                break
    ps_df = tmp_df[[self.match_id,self.target_var,self.treat_var]+current_best_vars]
    
    return(ps_df,current_best_vars) """


    with st.expander('##### Helper 4: generate_interactions'):
        st.markdown('''*This method takes first_order variables that are dummy coded and the default is to return all interactions and squared terms. Can specify which variables to keep but this needs to be handled in the parent function.*''')
        generate_interactions = """ def _generate_interactions(self,first_order,all_interactions = True, all_squared=True,include_interactions = None, include_squared = None):
            
    #Dictionary key that determines whether variable is continuous and only squares those it finds
    cont_check = {k:(False if self.masterDf[k].nunique()==2 and all(np.unique(self.masterDf[k])==np.array([0,1])) else True) for k in first_order}
    
    ##Checks if all variables specified in first order should be squared 
    if all_squared:
        all_squares_dict = {i: self.masterDf[i]*self.masterDf[i] for i in cont_check if cont_check[i]}
        #Create lists based on dict to combine with interactions
        square_terms = [v for v in all_squares_dict.values()]
        square_columns = [k+"_2" for k in all_squares_dict.keys()]
    ##Code for option all_squared is False
    else:
        if include_squared is not None:
            
            all_squares_dict = {i: self.masterDf[i]*self.masterDf[i] for i in cont_check if cont_check[i] and i in include_squared}
            if len(all_squares_dict)>0:
                square_terms = [v for v in all_squares_dict.values()]
                square_columns = [k+"_2" for k in all_squares_dict.keys()]
            else:
                square_columns = []
                square_terms = []
        else:
            square_columns = []
            square_terms = []
    
    #Generate all interactions
    if all_interactions:
        all_interactions = list(itertools.combinations(first_order,2))
        all_int_columns = [i[0]+"."+i[1] for i in all_interactions]
        
    else:
        if include_interactions is not None:
            
            all_interactions = list(itertools.combinations(include_interactions,2))
            all_int_columns = [i[0]+"."+i[1] for i in all_interactions]
        else:
            all_interactions = []
            all_int_columns = []
    
    if len(all_int_columns) + len(square_columns)==0:
        return pd.DataFrame()
    
    all_interactions = [self.masterDf[i[0]] * self.masterDf[i[1]] for i in all_interactions] if len(all_interactions)>0 else []
    #Create dataframe with squared terms and interactions
    interact_df = pd.DataFrame(np.vstack(square_terms + all_interactions).T,columns = square_columns + all_int_columns)
    
    ##Remove any interactions that have no variance and mostly gets rid of interactions with indicator variables
    all0 = [i for i in interact_df.columns if interact_df[i].nunique()==1]
    interact_df.drop(all0, axis=1, inplace=True)
    return interact_df"""

    with st.expander('##### Helper 5: get_all_indicators'):
        st.markdown('''*This takes as input an array-like object of column names and is intended to be used based on model name of self.best_model_meta. If is_binary is True then existing _vars must be in order of model name stored in best_model_meta. It will find all indicator variables of category variable if at least one is in the ps model. It returns those along with the numeric and interaction variables. It also returns a boolean vector to be used with is_binary in stat calculations.* ''')
        get_all_indicators = """def _get_all_indicators(self,existing_vars,is_binary=None,allow_refs=True):
    assert isinstance(existing_vars,(list, tuple, np.ndarray,pd.Series))
    existing_vars = pd.Series(existing_vars)
    if is_binary is not None:
        bin_dict = dict(zip(existing_vars,is_binary[3:-2]))
        #Split on interactions and explode
        existing_vars = existing_vars.str.split(".").explode()
        #Remove numeric vars
        existing_vars = existing_vars[existing_vars.str.startswith(tuple(self.cat_vars))]
        #Create the pattern matching
        cat_pat = "|".join(self.cat_vars)
        #Extract the stem from existing vars and remove duplicates
        existing_cat_vars = existing_vars.apply(lambda x: re.search(cat_pat,x).group()).unique()
        all_ind_vars = pd.Series(self.indicator_vars)
        all_ind_vars = all_ind_vars[all_ind_vars.str.startswith(tuple(existing_cat_vars))]
        if not allow_refs:
            all_ind_vars = list(np.setdiff1d(all_ind_vars, self.ref_vars))
        #Update bin dict with all 
        bin_dict.update({k:True for k in all_ind_vars})
        return bin_dict
    else:
        #Find and store interactions for final return
        interactions = list(existing_vars[existing_vars.str.contains("[.]",regex=True)])
        #Split on interactions and explode
        existing_vars = existing_vars.str.split(".").explode()
        #Find and store numeric vars for final return
        num_vars = list(existing_vars[existing_vars.str.startswith(tuple(self.num_vars))].unique())
        #Remove numeric vars
        existing_vars = existing_vars[existing_vars.str.startswith(tuple(self.cat_vars))]
        #Create the pattern matching
        cat_pat = "|".join(self.cat_vars)
        #Extract the stem from existing vars and remove duplicates
        existing_cat_vars = existing_vars.apply(lambda x: re.search(cat_pat,x).group()).unique()
        all_ind_vars = pd.Series(self.indicator_vars)
        all_ind_vars = all_ind_vars[all_ind_vars.str.startswith(tuple(existing_cat_vars))]
        if not allow_refs:
            all_ind_vars = list(np.setdiff1d(all_ind_vars, self.ref_vars))
        return sorted(all_ind_vars + num_vars + interactions) """

        st.code(get_all_indicators,language="python")
        
    with st.expander('##### Helper 6: calc_linear_psm'):
        calc_linear_psm = """def _calc_linear_psm(self, model_key, model_info, ps_arr):
    lpsm = np.nan
    try:
        lpsm = np.log(ps_arr/(1-ps_arr))
    except RuntimeWarning as e:
        self.logger.warning(f'The model {model_key} {model_info} encountered the following error: {e}')
    finally:
        return lpsm"""
        st.code(calc_linear_psm,language="python")
 

    st.header('Calculating the Propensity Score')
    st.markdown('#### Once the variables and second-order terms were identified, a logistic regression was used to create the propensity score by regressing HSV-1 on the selected covariates. Consistent with Imbens and Rubin we converted the estimated propensity scores to linearized propensity scores because matching on the linearized propensity score or the log odds ratio is more appropriate than the actual propensity score<sup class="sup">5</sup>. This is because a linearized score is more likely to have a distribution that can be more closely approximated by a normal distribution<sup class="sup">5</sup>. Helper function 6 above is used to calculate the linearized propensity score in the PSA class while selecting the covariates.',unsafe_allow_html=True)
    
    st.header('Calculating Baseline Metrics and Assessing Overlap')
    st.markdown('#### After creating a linearized propensity score, baseline metrics were used to access overlap between the treatment and control groups. We calculated the absolute difference in means between treatment and controls on each matching covariate and linearized propensity score.  Because there are usually more control units than treatment units, these absolute difference in means were scaled by the pooled standard deviations where each standard deviation was given equal weight<sup class="sup">6</sup>.  We used Austin’s formulas to calculate continuous (1a) and dichotomous(1b) covariates<sup class="sup">7</sup>. ',unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,1])
    col1.markdown('##### (1a)')
    col1.latex(r'''d = \left(\frac{\left| \bar{x}_{\text{treatment}} - \bar{x}_{\text{control}} \right|}{\sqrt{\frac{s_{\text{treatment}}^2 + s_{\text{control}}^2}{2}}}\right)''')
    
    col2.markdown('##### (1b)')
    
    col2.latex(r'''d = \frac{\left| \hat{p}_{\text{treatment}} - \hat{p}_{\text{control}} \right|}{\sqrt{\frac{\hat{p}_{\text{treatment}}(1-\hat{p}_{\text{treatment}}) + \hat{p}_{\text{control}}(1-\hat{p}_{\text{control}})}{2}}}''')
    
    
    st.markdown('#### Hypothesis tests and p-values were not used to consider balance because hypothesis tests often conflate changes in balance with changes in statistical power<sup class="sup">8,</sup><sup class="sup">9</sup>. Instead the log of the ratio of standard deviations was calculated because difference in logarithms is typically more normally distributed than the difference in their standard deviations<sup class="sup">6</sup> as shown below in (1c).',unsafe_allow_html=True)
    
    st.markdown('')
    st.markdown('')
    st.markdown('##### (1c)')
    st.latex(r'''\Gamma_{ct} = \ln \left( \frac{s_{\text{treatment}}}{s_{\text{control}}} \right)
''')

    st.markdown('#### The ratio of variances of the propensity scores were calculated between the treatment and control groups. Covariates with variance ratios greater than 2 or less than ½   were addressed in phase three during matching</sup><sup class="sup">10</sup>.  We also calculated the ratio of the variance of residuals orthogonal to the propensity score in the treated and control groups and these should also be between ½ and two</sup><sup class="sup">10</sup>.',unsafe_allow_html=True)
    
    st.markdown('#### To directly assess overlap, we calculated, for each covariate and the linearized propensity score the fraction of the treated units who fall outside of the 0.025 and 0.975 quantiles of the values of the control units</sup><sup class="sup">5</sup>. If this fraction is large, matching will be problematic because there will be very poor matches. This same fraction was calculated but with comparison units who fall outside the range of values of the treated units. ', unsafe_allow_html=True)
    
    st.markdown('#### In addition to these individual covariate metrics, we also calculated the overall balance. This overall balance is defined as the Mahalanobis distance between the means of the treatment and control units with respect to the pooled standard deviation. This formula is explained in greater detail in Imbens and Rubin</sup><sup class="sup">5</sup>. A score of 0 indicates that all covariates are exactly balanced.',unsafe_allow_html=True)
    
    st.markdown('#### Below is the code that is necessary to calculate all of the above metrics which establish a baseline for subsequent phases. The requires parameter takes an index number which corresponds to a covariate selection model obtained from the determine_ps_feats function described above: ')
    
    st.code('my_psm.assess_covariate_balance(1)')
    
    st.markdown("#### If you are interested to see the how the assess_covariate_balance method of the PSA class works, click below to view it:")
    
    with st.expander('##### Main Function: assess_covariate_balance'):
        st.markdown('''*This method outputs the baseline diagnostics of the function before matching*''')
        assess_covariate_balance = """ def assess_covariate_balance(self, index, custom = False, first_order_pval = stats.norm.sf(1), second_order_pval = 0.05, overlap_thresh=0.1, digits = 4, **plotly_kwargs):
    ##Check for plotly keyword args and if they are valid and return dictionary of valid key, val pairs
    dist_plot_kw = self._check_dist_plot_keys(plotly_kwargs)
    #Extract pval key based on user input
    if not custom:
        pval_key = self._convert_pval_key(first_order_pval,second_order_pval)
    else:
        pval_key = "custom"
    #_process_desc_stats function returns a 4 element tuple of a the new key generated, boolean whether using a different threshold value for previously run model, dataframe of basic diagnostics for each variable and scaler for MHD Balance metric
    model_key, thresh_only, basic_desc_stats, match_df = self._process_desc_stats(pval_key, index,overlap_thresh)
    ##Initialize model key for matched sets to avoid race conditions in create matches
    if self.matched_sets.get(model_key) is None:
        self.matched_sets[model_key] = {}
    #Check if match_df already been run!
    if match_df is None:
        return None
    
    #Print results to excel
    basic_results_dict = {'Before_matching_diagnostics': {'Overall Balance Statisics':match_df,'Descriptive Statistics': basic_desc_stats}}
    titles_diagnos = {'Before_matching_diagnostics': {'Overall Balance Statisics':f'Overall Balance Info (overlap threshold = {overlap_thresh})','Descriptive Statistics': 'Descriptive Statistics'}}
    self._print_excel_dict(basic_results_dict, f'{self.main_path}/{model_key}_thresh{overlap_thresh}_results.xlsx', table_titles = titles_diagnos, digits = digits)
    
    if thresh_only:
        self.desc_stats[model_key]['overall_balance_stats'].update({"threshkey"+str(overlap_thresh): match_df})
        ##Skip the plot if user does not specify and basic stats have already been run
        if len(dist_plot_kw)==0:
            return match_df
     
    #Case where basic stats have never been run    
    else:
    
        #Append dataframe and MHD scaler to desc_stats_model_dict and initialize dist_plot sub dictionary
        self.desc_stats[model_key].update({'basic_stats': basic_desc_stats,
                                           'overall_balance_stats': {"threshkey"+str(overlap_thresh): match_df},
                                           'dist_plot':{'lpsm':{},'psm':{}}})
    ###Create the dist_plot
    std_multiple = dist_plot_kw.get('std_multiple')
    ##Set default bin size to 0.1 of a pooled standard deviation
    if std_multiple is None:
        dist_plot_kw['std_multiple'] = 0.1
    
    ###This is where the dist plot happens - needs df that has been formatted with comparison, treatment   
    hist_fig = self.plotly_dist_plot(model_key,**dist_plot_kw)
    if hist_fig is None:
        self.logger.warning(f'No variation in the propensity scores for {model_key} due to extreme values')
    
    return hist_fig"""
        
        st.code(assess_covariate_balance,language='python')
    
    with st.expander('##### Helper 1: convert_pval_key'):
        convert_pval_key = """def _convert_pval_key(self, first_order_pval, second_order_pval):
    ##Create pval_key based on user entered criteria for decision rules. If value is non-terminating round
    if len(str(first_order_pval))-2>4:
        pval_key = f'thresh {first_order_pval:.4f},'
    else:
        pval_key = f'thresh {first_order_pval},'
    
    if not second_order_pval:
        return pval_key
    
    if len(str(second_order_pval))-2>4:
        pval_key = pval_key+f'{second_order_pval:.4f}'
    else:
        pval_key = pval_key + str(second_order_pval)
    return pval_key """
        
        st.code(convert_pval_key,language='python')
        
    
    with st.expander('##### Helper 2: process_desc_stats'):
        st.markdown( '''*This function extract data frame based on pval_key and index number and then returns the baseline descriptive stats or None if it already has been run.*''')
        process_desc_stats = """def _process_desc_stats(self,pval_key, index,overlap_thresh,post_match_assess = False,sing_mat_corr = 1e-6, allow_refs = True):
            thresh_only = False
            current_index = len(self.desc_stats)
            #Extract model name from best_model_meta and Extract is_binary indicator for each column in order to apply appropriate standard deviation formula
            model_name, is_binary = self.best_model_meta[pval_key][index]
            #Checking for duplicates!
            if current_index>0:
                #Extract the name key for model info
                #If already run then return None
                for k, v in self.desc_stats.items():
                    #If key exists then check further 
                    if v.get("name")==model_name:
                        old_thresh_keys = v.get('overall_balance_stats').keys()
                        if "threshkey"+str(overlap_thresh) in old_thresh_keys:
                            self.logger.warning(f'You have already run diagnostics for the dataframe with a threshold of {overlap_thresh} for the following covariates: {model_name}. Choose another model or threshold value to use this method. If you wish to only create a dist plot use the plotly_dist_plot method!')
                            return "threshkey"+str(overlap_thresh), thresh_only, None, None
                        thresh_only = True
                        old_key = k
                        break
            
            new_key = "psmodel" + str(current_index)
            #Extract ps_df based on pval_key and index and drop match_id and outcome variable
            custom = True if pval_key=='custom' else False
          
            ps_df = self.get_psm_df(index, custom=custom, manual_pval_key=pval_key).drop([self.match_id,self.target_var], axis = 1).copy()

            ##Splitting ps_df into a comp and tr set
            comp_arr, tr_arr = self._calc_np_groupby_split(ps_df.to_numpy(), 0)
            
            ##Remove the grouping column (i.e, treatvar)
            comp_arr = np.take(comp_arr, np.arange(1,comp_arr.shape[1]),axis=1)
            tr_arr = np.take(tr_arr, np.arange(1,tr_arr.shape[1]),axis=1)
            ####For overall_balance_stats df####
            ##Overlap metric
            c_psvals, c_lpsvals = np.take(comp_arr,[-2,-1],axis=1).T
            tr_psvals, tr_lpsvals = np.take(tr_arr,[-2,-1],axis=1).T        

            tr_lp_overlap = self._calc_ps_overlap(tr_lpsvals,c_lpsvals,overlap_thresh)
            tr_p_overlap = self._calc_ps_overlap(tr_psvals,c_psvals,overlap_thresh)
            c_lp_overlap = self._calc_ps_overlap(c_lpsvals,tr_lpsvals,overlap_thresh)
            c_p_overlap = self._calc_ps_overlap(c_psvals,tr_psvals,overlap_thresh)
            
            ##Calculate group means
            comp_mean = comp_arr.mean(axis=0)
            tr_mean = tr_arr.mean(axis=0)
            
            ##MHD balance metric
            ##Need to remove last 2 which include the ps vars
            stat_type = "processing descriptive statistics"
            covar_only_indices = np.arange(0,(len(comp_mean)-2))
            mhd_bal = self._calc_mhd_balance(new_key, stat_type, np.take(comp_arr,covar_only_indices,axis=1),np.take(tr_arr,covar_only_indices, axis=1),np.take(comp_mean,covar_only_indices),np.take(tr_mean,covar_only_indices),sing_mat_corr)
            
            ###Combine matching statistics into separate table
            match_stat_var_names = ["Overall Multivariate Difference in Covariates","% feasible matches for treatment (linearized propensity score)", "% feasible matches for comparison (linearized propensity score)","% feasible matches for treatment (propensity score)", "% feasible matches for comparison (propensity score)"]
            match_stat_values = np.array([mhd_bal,tr_lp_overlap,c_lp_overlap,tr_p_overlap,c_p_overlap])
            match_stat_df = pd.DataFrame(zip(match_stat_var_names, match_stat_values),columns = ['Properties','Value'])
            
            if thresh_only:
                basic_desc_stats = self.desc_stats[old_key]['basic_stats']
                return old_key, thresh_only, basic_desc_stats, match_stat_df
            else:
                #Instantiate new nested dict
                self.desc_stats.update({new_key: {'pval_key': pval_key, 
                                                  'df_index': index,
                                                  'name':model_name}})
            
            ##Calculate group std according to categorical versus continuous
            #Get is_binary from meta data to indicate which column is binary (Need to start at 3 bc match_id, target,and treat var)
            n_cols = len(is_binary)
            is_binary = np.take(is_binary,np.arange(3,n_cols))
            model_info = 'during calculation of standard deviation for processing desciptive statistics'
            model_key=new_key
            std_func = partial(self._calc_std,model_key, model_info)
            comp_sd = np.array(list(itertools.starmap(std_func,zip( comp_arr.T,is_binary,comp_arr.mean(axis=0)))))
            tr_sd = np.array(list(itertools.starmap(std_func,zip(tr_arr.T,is_binary,tr_arr.mean(axis=0)))))
            
            #Calculate counts
            comp_N = np.array(list(itertools.starmap(self._calc_count, zip(comp_arr.T, is_binary))))
            tr_N = np.array(list(itertools.starmap(self._calc_count, zip(tr_arr.T, is_binary))))
            
            #Calculate quantiles for 95% interval
            comp_q025 = np.array(list(map(self._calc_quant025,comp_arr.T)))
            tr_q025 = np.array(list(map(self._calc_quant025,tr_arr.T)))
            
            comp_q975 = np.array(list(map(self._calc_quant975,comp_arr.T)))
            tr_q975 = np.array(list(map(self._calc_quant975,tr_arr.T)))
            
            #Calculate coverage frequency
            comp_coverage = np.array(list(map(self._calc_coverage_freq,comp_arr.T)))
            tr_coverage = np.array(list(map(self._calc_coverage_freq,tr_arr.T)))
            
            model_key=new_key
            ##Normalized Differences
            model_info = 'during calculation of normalized difference in means for processing descriptive statistics'
            norm_diff_func = partial(self._calc_norm_diff, model_key, model_info, None)
            norm_diff = np.array(list(itertools.starmap(norm_diff_func, zip(tr_mean, comp_mean, tr_sd, comp_sd))))
           
            ##Ratio of variances
            model_info = 'during calculation of ratio of variances for processing descriptive statistics'
            var_ratio_func = partial(self._calc_var_ratio, model_key, model_info)
            var_ratio = np.array(list(itertools.starmap(var_ratio_func, zip(tr_sd, comp_sd))))
            
            ##Ratio of residual variances
            
            #Remove the ps columns since those are the covars
            stat_type = "processing descriptive statistics"
            resid_vars = np.arange(0,tr_arr.shape[1]-2)
            #Get ratio for ps resids
            tr_ps = np.take(tr_arr,-1,axis=1)
            c_ps = np.take(comp_arr,-1,axis=1)
            resid_func = partial(self._get_ps_resid_var_ratios,model_key,stat_type, tr_ps,c_ps)
            resid_var_ps = np.array(list(itertools.starmap(resid_func, zip(np.take(tr_arr,resid_vars,axis=1).T, np.take(comp_arr,resid_vars,axis=1).T,np.take(is_binary,resid_vars)))))
            #Pad the missing values for psm with nan
            resid_var_ps = np.concatenate([resid_var_ps,np.repeat(np.nan,2)])
            
            ##Ratio of log std
            model_info = 'during calculation of log of the standard deviations for processing desciptive statistics'
            
            log_std_ratio_func = partial(self._calc_log_std_ratio,model_key,model_info)
            log_sd_ratio = np.array(list(itertools.starmap(log_std_ratio_func, zip(tr_sd,comp_sd))))
           
            stat_colnames = ["Covariates", "Norm_diff", "Var_ratio", "Resid_ps_var_ratio","Log_SD_ratio", "Coverage05_c", "Coverage05_tr", "N_c","N_tr","Mean_c", "Mean_tr","SD_c", "SD_tr","Q025_c",	"Q025_tr",	"Q975_c",	"Q975_tr"]
            
            basic_desc_stats = pd.concat([pd.Series(ps_df.columns[1:]),pd.DataFrame(np.c_[norm_diff, var_ratio, resid_var_ps,log_sd_ratio, comp_coverage, tr_coverage, comp_N, tr_N, comp_mean, tr_mean, comp_sd, tr_sd, comp_q025, tr_q025, comp_q975, tr_q975])],axis=1)
            basic_desc_stats.columns = stat_colnames
            
            ##Convert the N columns to integer for better report
            basic_desc_stats['N_tr'] = basic_desc_stats['N_tr'].astype(object)
            basic_desc_stats['N_c'] = basic_desc_stats['N_c'].astype(object)
            basic_desc_stats = basic_desc_stats.sort_values('Norm_diff',ascending=False).reset_index(drop=True)
            
            return new_key, thresh_only,pd.concat([basic_desc_stats,pd.DataFrame([["Multivariate measure", mhd_bal]],columns = ["Covariates","Norm_diff"])]).reset_index(drop=True) , match_stat_df """
        
        st.code(process_desc_stats,language="python")
        
        
    with st.expander('##### Helper 3: print_excel_dict'):
        st.markdown('''*This function is a generic helper function for printing the results to excel. It requires a two-level dictionary with outer keys as subsets and inner keys as pipeline keys. Titles must all follow the same structure and functions calling this will create this before calling this function*''')
        print_excel_dict = """ def _print_excel_dict(self,twolevel_dict,file_path,table_titles = None, digits=4,match_meta_add_on = False):
    writer = pd.ExcelWriter(file_path,engine= 'xlsxwriter')
    border_color = "red"
    #Top level of loop is subset level so each excel sheet corresponds to different subsets
    for subset, subset_dict in twolevel_dict.items():
        sheetvals = []
        model_names = []
        meta_vals = []
        for model_key, df in subset_dict.items():
            sheetvals.append(df)
            model_names.append(model_key)
            if match_meta_add_on:
                if not (subset.startswith("Match Comparison") or model_key.startswith("Before")):
                    meta_vals.append(self._print_add_on_match_meta(model_key))
                else:
                    meta_vals.append(None)
            else:
                meta_vals.append(None)
                
        #Loop through all sections consisting of df's
        start_col = 0
        for i, df in enumerate(sheetvals):
            #Initialize to create a sheetname to attach to workbook
            section_len = df.shape[1]+1 #Separate each section by one column
            pd.DataFrame([" "]).to_excel(writer, sheet_name= subset,startrow=0,startcol= start_col, index=False,header=False)
            #Create workbook
            workbook= writer.book
            worksheet = writer.sheets[subset]
            cell_format = workbook.add_format({'bold': True, 'font_color': 'red', 'font_size':16})
            cell_format_center = workbook.add_format({'bold': True, 'font_color': 'red', 'font_size':16, 'align': 'center'})
            cell_format_meta_match = workbook.add_format({'bold': True, 'font_color': 'black', 'font_size':16, 'align': 'center','bottom':6, 'bottom_color':border_color})
            cell_format_wrap_left = workbook.add_format({'text_wrap': True, 'align': 'center','valign': 'left','left':6, 'left_color':border_color,'right_color': border_color})
            cell_format_wrap_right = workbook.add_format({'text_wrap': True, 'align': 'left','valign': 'center','right': 6, 'right_color': border_color})
            cell_format_wrap_left_bottom = workbook.add_format({'text_wrap': True, 'align': 'left','valign': 'left','left':6,'bottom':6, 'bottom_color':border_color,'left_color':border_color})
            cell_format_wrap_right_bottom = workbook.add_format({'text_wrap': True, 'align': 'center','valign': 'center','right': 6,'bottom':6, 'bottom_color':border_color,'right_color':border_color})
            cell_format_wrap_left_header = workbook.add_format({'text_wrap': True, 'align': 'center','valign': 'left','left':6, 'left_color':border_color,'right_color': border_color,'bold':True})
            cell_format_wrap_right_header = workbook.add_format({'text_wrap': True, 'align': 'left','valign': 'center','right': 6, 'right_color': border_color,'bold':True})
            
            #Write the sections
            merge_len = section_len - 2 #This is effectively the number of columns -1 for 0 indexing
            if table_titles is None:
                section_title = model_names[i]
            else:
                section_title = table_titles[subset][model_names[i]]
            worksheet.merge_range(0,start_col,0,start_col+merge_len,section_title, cell_format_center)
            
            mod_df = df.copy()
            col_width_format = []
            for idx, col in enumerate(mod_df.columns):  # loop through all columns
                series = mod_df[col]
                if series.dtypes in ['float64']:
                    #series = np.round(series,digits)
                    series = series.apply(lambda x: np.nan if np.isinf(x) else x)
                    if digits is not None:
                        series = series.apply(lambda x: f'{{:,.{digits}f}}'.format(x))
                        mod_df[col] = series
                        
                max_len = max((series.astype(str).map(len).max(),  # len of largest item
                               len(str(series.name))  # len of column name/header
                               )) + 0.5  # adding a little extra space
                col_width_format.append((f'{xl_col_to_name(start_col+idx)}:{xl_col_to_name(start_col+idx)}', max_len))
            
            mod_df.to_excel(writer, sheet_name= subset,startrow=1,startcol= start_col, index=False)
                  
            for col, max_len in col_width_format:
                if col=='A:A' and meta_vals[i] is not None:
                    worksheet.set_column(col, max(max_len, 25)) # set column width
                else:
                    worksheet.set_column(col, max_len)
            start_col = start_col + section_len
            
            ###This code adds the formatting for the meta results and must be applied after setting column widths for valign to work I think
            if meta_vals[i] is not None:
                worksheet.set_column(1, 15)  # set column width of meta match
                meta_col_headers = list(meta_vals[i].columns)
                ##Need to find the number of rows in dataframe + 2 for the column names and header row and then one more after that. Maybe 2 to be safe!
                match_meta_start_row = len(mod_df)+2 + 2
                worksheet.merge_range(match_meta_start_row ,0, match_meta_start_row, 1 ,"Matching Meta-Data", cell_format_meta_match)
                meta_vals[i].to_excel(writer, sheet_name = subset, startrow = match_meta_start_row+1, startcol = 0, index = False)
               
                ##Make side borders so skip first row which is exact and also the bottom row because it needs two borders
                worksheet.write(match_meta_start_row+1, 0,meta_col_headers[0], cell_format_wrap_left_header)
                worksheet.write(match_meta_start_row+1, 1,meta_col_headers[1], cell_format_wrap_right_header)
                for j, row in enumerate(meta_vals[i].itertuples()):
                    
                    worksheet.write(match_meta_start_row+2+j, 0,row[1],cell_format_wrap_left)
                    worksheet.write(match_meta_start_row+2+j, 1,row[2],cell_format_wrap_right)
                #Draw border on bottom row
                last_row_meta = meta_vals[i].iloc[len(meta_vals[i])-1]
                worksheet.write(match_meta_start_row+1+len(meta_vals[i]), 0,last_row_meta[0],cell_format_wrap_left_bottom)
                worksheet.write(match_meta_start_row+1+len(meta_vals[i]), 1,last_row_meta[1],cell_format_wrap_right_bottom)
    
    writer.save()
    return None"""
        
        st.code(print_excel_dict, language="python")
        
    with st.expander('##### Helper 4: plotly_dist_plot'):
        plotly_dist_plot = """def plotly_dist_plot(self, model_key, post_match = None, match_key = None, psm_type = "lpsm", density_type = "kde", binwidth = 0.5, std_multiple = 0, height = 1500, width = 1000, treat_color = "lightgreen", comp_color = "magenta", plot_bgcolor = "black", shade = True, shade_fillcolor = "white", spikecolor = "yellow", barmode='group',linetype = 'dot'):
            
    assert linetype in ['dot','dash','dashdot',None]
    ##Pass the value checker for psm_type, density_type, binwidth, std_multiple
    args = locals()
    plot_kw_check = self._check_dist_plot_keys(args)
    
    ##Extract meta data from desc_stats to access the appropriate data frame (assess covariates must be run prior to plot!!!)
    if post_match is None:
        pval_key = self.desc_stats[model_key].get('pval_key')
        index = self.desc_stats[model_key].get('df_index')
        
        custom = True if pval_key=='custom' else False
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist_df = self.get_psm_df(index, custom = custom, manual_pval_key = pval_key, allow_refs = False)
    
    else:
        assert isinstance(post_match,pd.DataFrame)
        hist_df = post_match
        
    hist_df[self.treat_var] = np.where(hist_df[self.treat_var]==1,"Treatment","Comparison")
    
    ##Extract the quantiles for shading and focus on extreme values
    basic_stats = self.desc_stats[model_key]['basic_stats'].copy()
    
    #Create the indicator for linearized versus nonlinearized
    psm_print_name = "Linearized Propensity Score" if psm_type=="lpsm" else "Propensity Score"
    psm_type_df = "lpsm_score" if psm_type=='lpsm' else "psm_score"
     
    lower_quant_bound = basic_stats.loc[basic_stats.Covariates==psm_type_df,"Q025_tr"].values[0]
    higher_quant_bound = basic_stats.loc[basic_stats.Covariates==psm_type_df,"Q975_c"].values[0]
     ##Calculate the binwidth based on multiple of standard deviation
    if std_multiple>0:
        std_c, std_t = basic_stats.loc[basic_stats.Covariates==psm_type_df,["SD_c","SD_tr"]].values.T
        binwidth = std_multiple*np.sqrt((std_t**2 + std_c**2)/2)[0]
    
    ##Round to 3 places for displaying binwidth size in title - Scientific notation for less than that
    if binwidth<=0.0005:
        binwidth_print = f'{{:1.1e}}'.format(binwidth)
    else:
        binwidth_print = str(np.round(binwidth,3))
    
    ##Creating subsets for generating bin counts and bin width
    treat = hist_df.loc[hist_df[self.treat_var]=="Treatment"].copy()
    comparison = hist_df.loc[hist_df[self.treat_var]=="Comparison"].copy()
    
    ##Set min and max for compatability with np.histogram
    bin_start = np.floor(hist_df[psm_type_df].min())
    bin_end = np.ceil(hist_df[psm_type_df].max())
    
    ##Checks to see if no variation in the propensity scores because histogram not possible then!
    if bin_end-bin_start==0:
        return None
   
    nbins, treat_count_tup, comp_count_tup=  self._get_plotly_hist_counts([treat[psm_type_df],comparison[psm_type_df]], bin_start, bin_end, binwidth)
    
    if density_type is not None:
        assert density_type in ["kde","normal"]
        treat_x_range = treat_count_tup[1] + (binwidth/2)
        comp_x_range = comp_count_tup[1] + (binwidth/2)
        if density_type=='kde':
            try:
                treat_kde = stats.gaussian_kde(treat[psm_type_df])
                comp_kde = stats.gaussian_kde(comparison[psm_type_df])
            except LinAlgError as e:
                self.logger.warning(f'The following LinAlgError arose in calculating the kde curve for {model_key}_{match_key}: "{e.args[0]}". Defaulting to normal curve only')
                density_type='normal'

    ##Plot overall histogram normalized with probability density 
    fig = px.histogram(hist_df, x=psm_type_df,color = self.treat_var,marginal="rug",histnorm='probability density', histfunc = "count", nbins=nbins, hover_name="match_id",labels = {psm_type_df:psm_print_name,self.treat_var:""}, barmode=barmode, color_discrete_map = {"Comparison":comp_color,"Treatment":treat_color}, template="plotly")
    
    ###Add kernel density if requested to histogram
    if density_type=='kde':
        
        fig.add_trace(go.Scatter(x = comp_x_range,
                                   y = comp_kde.evaluate(comp_x_range),
                                  mode = "markers+lines",name="KDE Curve (C)",
                                  line = dict(color = comp_color,
                                              dash = linetype
                                             ),
                                   marker = dict(color = comp_color),
                                  hovertemplate='<b> x: %{x:.2f}, y: %{y:.2f}',
                                xhoverformat = ".2f",
                               ))


        fig.add_trace(go.Scatter(x = treat_x_range,
                                   y = treat_kde.evaluate(treat_x_range),
                                  mode = "markers+lines",name="KDE Curve (Tr)",
                                  line = dict(color = treat_color,
                                              dash = linetype
                                             ),
                                   marker = dict(color = treat_color),
                                  hovertemplate='<b> x: %{x:.2f}, y: %{y:.2f}',
                                xhoverformat = ".2f",
                                ))

    ###Add normal curve to histogram if requested
    if density_type=='normal':
        
        ##Normal curve for comparison
        fig.add_trace(go.Scatter(x = comp_x_range,
                                 y = stats.norm.pdf(comp_x_range, np.mean(comparison[psm_type_df]), np.std(comparison[psm_type_df])),
                                 mode = "markers+lines",name="Normal Curve (C)",
                                  
                                 line = dict(color = comp_color, dash = linetype),
                                 marker = dict(color=comp_color),
                                 hovertemplate='<b> x: %{x:.2f}, y: %{y:.2f}',
                                 xhoverformat = ".2f",
                                ))
        ##Normal Curve for treatment
        fig.add_trace(go.Scatter(x = treat_x_range,
                                 y = stats.norm.pdf(treat_x_range, np.mean(treat[psm_type_df]), np.std(treat[psm_type_df])),
                                 mode = "markers+lines",name="Normal Curve (Tr)",
                                 line = dict(color = treat_color, dash = linetype),
                                 marker = dict(color = treat_color),
                                 hovertemplate='<b> x: %{x:.2f}, y: %{y:.2f}',
                                 xhoverformat = ".2f",
                                ))
    if shade:
        ##Right tail shading 
        fig.add_vrect(x0 = hist_df[psm_type_df].min(), x1= lower_quant_bound,
                        line_color = "lightskyblue",
                        line_dash = "dot",
                        fillcolor = shade_fillcolor,
                        opacity = 0.5)
        
        ##Left tail shading
        fig.add_vrect(x0 = higher_quant_bound, x1= hist_df[psm_type_df].max(),
                        line_color = "lightskyblue",
                        line_dash = "dot",
                        fillcolor = shade_fillcolor,
                        opacity = 0.5)

    ##Comparison histogram updates
    fig.update_traces(marker_opacity = 0.7,
                      xbins = dict(start = bin_start, end = bin_end, size=binwidth),
                      customdata =comp_count_tup[0],
                      hovertemplate = '<b>Bin Range: %{x}, Count: %{customdata} </b>',
                      marker_line = dict( color = 'black',width = 2),
                      selector = dict(type="histogram",offsetgroup = "Comparison"))
    
    #Treatment histogram updates
    fig.update_traces(marker_opacity = 0.7,
                      xbins = dict(start = bin_start, end = bin_end, size=binwidth),
                      customdata =treat_count_tup[0],
                      hovertemplate = '<b>Bin Range: %{x}, Count: %{customdata} </b>',
                      marker_line = dict( color = 'black',width = 2),
                      selector = dict(type="histogram",offsetgroup = "Treatment"))
    
    ##Comparison Rug plot update
    fig.update_traces(customdata =hist_df['match_id'],
                      hovertemplate = f'<b> Match ID: %{{customdata}} <br> {psm_print_name}: %{{x:.4f}} </b><extra></extra>',
                      xhoverformat = ".4f",
                      marker = dict(symbol = "line-ns-open"),
                      jitter = 0.6,
                      opacity = 1,
                      hoverlabel_bgcolor = comp_color,
                      hoverlabel_bordercolor = comp_color,
                      selector = dict(type="box",offsetgroup="Comparison"))
    
    ##reatment Rug plot update
    fig.update_traces(customdata =hist_df['match_id'],
                      hovertemplate = f'<b> Match ID: %{{customdata}} <br> {psm_print_name}: %{{x:.4f}} </b><extra></extra>',
                      xhoverformat = ".4f",
                      marker = dict(symbol = "line-ns-open"),
                      jitter = 0.6,
                      opacity = 1,
                      hoverlabel_bgcolor = treat_color,
                      hoverlabel_bordercolor = treat_color,
                      selector = dict(type="box",offsetgroup="Treatment"))
    
    
    ##Y-axis update for histogram (not rug)
    fig['layout']['yaxis'].update(dict(titlefont = dict(size=14, color='black'),
                                         title_text="Probability Density",
                                         tickfont_family = "Ariel Black",
                                        showspikes = True, 
                                         spikecolor=spikecolor,
                                        spikethickness=1,
                                        spikedash = "longdash",
                                        #spikesnap="cursor"
                                        ))
    ##X-axis update for histogram (not rug)
    fig['layout']['xaxis'].update(dict(titlefont = dict(size=14,color='black'), 
                                        tickfont_family = "Ariel Black",
                                        spikecolor =spikecolor,
                                        spikethickness=1,
                                        spikedash = "longdash",
                                        #spikesnap="cursor"
                                        ))
    ##Overall Layout Updates
    fig.update_layout(legend_font = dict(size=11,family = "Ariel Black"),
                      legend_traceorder = "reversed",
                      margin_r = 175, 
                      hovermode="x unified", 
                      width = width, 
                      height = height,
                      plot_bgcolor=plot_bgcolor,
                      title = dict(text = f"<b>Distribution Plot of {psm_print_name}s with outer 5% regions highlighted <br>(Binwidth={binwidth_print})</b>",
                                   font_size=20, x = 0.5,font_family = "Ariel Black"))
    return fig """
        
        st.code(plotly_dist_plot,language="python")
    
    
    
if selected=='Phase 2':
    st.title("Phase 2: Creating Several Matched Sets")
    
    st.markdown('#### We tried several matching techniques that have been used in practice over the past 40 years.  Each matching technique requires a distance metric which supplies the matching criteria. We chose exact, Mahalanobis and Euclidean distances that were calculated using the values of all the covariates or based entirely on the linearized propensity score<sup class="sup">3</sup>.  While exact matching is ideal, it often becomes infeasible if there are many matching covariates leading to several unmatched units; excluding many observations leads to larger bias<sup class="sup">3,</sup><sup class="sup">11</sup>.',unsafe_allow_html=True)
    
    st.markdown('#### Both a greedy algorithm and a dynamic algorithm were used to produce matched sets based on distances described above. Nearest neighbor matching method, a greedy algorithm that uses the closest available match without replacement. This was also done with and without a 0.2 caliper. A 0.2 caliper is often taken as a multiple of the standard deviation of the propensity score which the literature recommends being around 20%</sup><sup class="sup">6</sup>. We also used partial exact matching on APOE, ancestry, and body tissue source which forced exact matching on those covariates but allowed mismatches on other covariates.',unsafe_allow_html=True)
    
    
    st.markdown('#### *Click on the link below to see the main function for creating matches which allows for both greedy and optimal matching:*')
    
    with st.expander('##### Main Function: create_matches'):
        st.markdown('''*This is the main method of creating matches and this function calls lower functions corresponding to the options specified by the user!*''')
        create_matches = """ def create_matches(self, model_key, exact_vars = None, match_method = "nearest", replace=False, distance = "psm", distance_metric_params= None, psm_type = "lpsm",n_matches = 1,caliper_std = -1,nearest_start_method = 'min', exact_distance = 'euclidean', manual_caliper_distance = None, use_psm_other=False,full_matching = False, n_matches_min =1, n_matches_max=2, integral_precision=5):
            
    ##Extract meta data from desc_stats to access the appropriate data frame (assess covariates must be run prior to plot!!!)
    all_params = locals()
    self.logger = self._setup_logger()
    meta_params = {k:v for k,v in all_params.items() if k not in ['self','model_key','match_method']}
    
    if match_method=="nearest":
        debugger_params = {k:v for k,v in all_params.items() if k in ['match_method','distance','exact_distance','exact_vars','caliper_std','manual_caliper_distance','replace','n_matches']}
    elif match_method=="optimal":
        debugger_params = {k:v for k,v in all_params.items() if k in ['match_method','distance','exact_distance','exact_vars','caliper_std','manual_caliper_distance','full_matching','n_matches_min','n_matches_max']}
    
    pval_key = self.desc_stats[model_key].get('pval_key')
    index = self.desc_stats[model_key].get('df_index')
    
    custom = True if pval_key=='custom' else False
    
    ps_df = self.get_psm_df(index,custom=custom, manual_pval_key = pval_key).copy()
    
    ##Check to see if ranks are to be used as in rank_mahalanobis
    if distance=='rank_mahalanobis':
        ps_df = self._process_rank_mahal_df(ps_df,use_psm_other)

    #Split into treatment and comparison
    comp,treat = [ps_df.loc[ps_df[self.treat_var]==i] for i in [0,1]]
    
    ######Need to process exact matching variables if specified by user
    ##Check if exact variables is a list and is specified in dataframe
    if exact_vars is not None:
        invalid_exact = ['match_id',self.target_var, self.treat_var, 'lpsm_score', 'psm_score']
        ##Case where exact is specified as a string (either all or name of single column)
        if isinstance(exact_vars, str):
            if exact_vars=='all' or exact_vars in ps_df.columns:
                if exact_vars=='all':
                    exact_vars = [i for i in ps_df.columns if i not in invalid_exact]
                
                else:
                    if exact_vars in ps_df.columns:
                        exact_vars = self._check_user_category_vars([exact_vars],allow_refs=True)
            else:
                self.logger.exception(f'Exact parameter specified as a string must be either "all" or be a column in {[i for i in ps_df.columns if i not in invalid_exact]}')
                raise ValueError
        ##Case where user specifies a number - it must correspond to column name!
        elif isinstance(exact_vars, (int,float)):
            if exact_vars in ps_df.columns:
                exact_vars = self._check_user_category_vars([exact_vars],allow_refs=True)
            else:
                self.logger.exception(f'Exact parameter specified as number but refer to the name of column not index! Valid columns are one or more of {[i for i in ps_df.columns if i not in invalid_exact]}')
                raise ValueError
        
        ##Case where exact is specified as array-like
        elif isinstance(exact_vars, (tuple, list,np.ndarray)):
            #Performs check and returns the indicator variables if the parent variable is specified by user
            exact_vars = self._check_user_category_vars(exact_vars,allow_refs=True)
            #Check to see if 
            exact_check = [col for col in exact_vars if col in invalid_exact]
            if len(exact_check)>0:
                exact_vars=np.setdiff1d(exact_vars, exact_check)
                self.logger.warning(f'Cannot perform exact matching for {model_key} under the matching critera: {debugger_params} on the following: {invalid_exact} because it is either the ID variable, treatment variable or target variable. Removing from list of exact matching variables!')
        ##Invalid argument for exact parameter
        else:
            self.logger.exception(f'Invalid data type specified for the argument in the "exact" parameter for {model_key} under matching criteria: {debugger_params}! Must be a string or list consisting of one or more of the valid column names: {[i for i in ps_df.columns if i not in invalid_exact]}. Use "all" if you wish to perform exact matching on all variables and/or set argument for manual_caliper_distance = 0 for exact matching on the propensity score!')
            raise ValueError
        
        ###Assign clean exact vars to the meta_params to pass to matching function
        #Check if possible indicator var not in psm_df because user entered parent cat var!
        exact_vars = [col for col in exact_vars if col in ps_df.columns]
        meta_params.update({'exact_vars': exact_vars})
        
    ##Check that all parameters are valid
    assert isinstance(replace, bool)
    assert distance in BallTree.valid_metrics + ['psm','rank_mahalanobis']
    assert exact_distance in BallTree.valid_metrics + ['psm']
    assert nearest_start_method in ["min","random","max","asis"]
    assert psm_type in ['lpsm','psm']
    assert isinstance(caliper_std,(int,float))
    if manual_caliper_distance is not None:
        assert isinstance(manual_caliper_distance,(int,float))
    
    ##Check if valid n_matches for no replacement
    assert isinstance(n_matches, int) and (n_matches>0 or n_matches==-1)
    if n_matches>1 and not replace:
        max_match_ratio = int(len(comp)/len(treat))
        if n_matches>max_match_ratio:
            self.logger.exception(f'There are not enough observations for {n_matches}:1 matches without replacement! Either use n_matches in the range of [1,{max_match_ratio}] or try matching with replacement by setting replace =True')
            raise ValueError
    
    G = None
    full_to_var_match = False
    new_fullmatch_meta_params = None
    if match_method=="nearest":
        
        match_results_tuple = self._match_nearest_neighbor(model_key,comp, treat, **meta_params)
        
        if match_results_tuple is None:
            return None
        
        matches, unmatched, final_meta_params, new_key = match_results_tuple
       
        if len(matches)==0:
           self.logger.warning(f'Your current caliper resulted in no matches for {model_key} with matching critera {debugger_params}. Either make your caliper less strict and/or do not use exact matching! This model will be included in matched_sets meta data under key {new_key} to prevent this matching scenario from running in the future.')
        
    elif match_method == "optimal":
        
        #This is first optimal run and will return None if original was duplicate but could also have full_to_var_match indicator if full matching was run bc initial ratio match was infeasible. Could also have empty matches if it failed and needs to be updated in matched_sets
        match_results_tuple = self._optimal_matching(model_key, comp, treat, **meta_params)
        
        #Only if duplicates and if full match was called for but failed. Could also be if BallTree fails and no optimal network could be contructed
        if match_results_tuple is None:
            return None
        
        #Unpack tuple and check for full_to_var_match. If full match was called for and failed then exact_matches is tup is None, and full_to_var_match is False
        matches, unmatched, final_meta_params, G, new_key, full_to_var_match, exact_matches_tuple = match_results_tuple

        if full_to_var_match:
            #Check if G is none because that happens if full match already run to help with variable matching
            if G is None:
                ##If full match already existed but won't work because empty
                if len(matches)>0:
                    #This function runs if we got old legit full match results to pass in
                    full_to_var_match_tup = self._optimal_full_to_var_match(model_key, matches.to_numpy(), n_matches_min, n_matches_max, exact_matches_tuple, final_meta_params, unmatched)
                    
                    matches, unmatched, final_meta_params, G = full_to_var_match_tup
                
        #This means full match was never run so result of optimal_matching function is full match rather than variable matching! But could also fail!
            else:
                if len(matches)>0:
                    
                    #This means we had to run full match and it worked! If not then matches is 0. Need to update these at end!
                    new_fullmatch_meta_params = {'df':matches,'unmatched_tr':pd.unique(unmatched), **final_meta_params}
                   
                    #Need to modify meta_params that get passed to optimal_match before turned into full match
                    final_meta_params_mod = copy.deepcopy(final_meta_params)
                    final_meta_params_mod.update({'n_matches_min': n_matches_min,
                                                  'n_matches_max': n_matches_max,
                                                  'optimal_matching_ratio':'constant' if n_matches_min==n_matches_max else 'variable'})
                    
                    full_to_var_match_tup = self._optimal_full_to_var_match(model_key, matches.to_numpy(), n_matches_min, n_matches_max, exact_matches_tuple, final_meta_params_mod, unmatched)
                    
                    
                    matches, unmatched, final_meta_params, G = full_to_var_match_tup
            
    #This updates match results with
    ###This is the main loop where everything gets updated including full_to_var_match and nearest
    tmp_match_meta_params = {'df':matches,'unmatched_tr':pd.unique(unmatched), **final_meta_params}
    try:
        
        self.lock.acquire()
    #This means it is not a shared dictionary
    except AttributeError:
        ##Checks if also need to add optimal full
        if new_fullmatch_meta_params is not None:
            is_duplicated, new_full_key = self._check_match_dup(model_key,match_method,new_fullmatch_meta_params)
            if is_duplicated:
                self.logger.warning(f'You passed the same model twice in a parallel operation but the original can be found at {new_key}')
            else:
                self.matched_sets[model_key][new_full_key] = new_fullmatch_meta_params
        ##Actual match
        is_duplicated, new_key = self._check_match_dup(model_key,match_method,tmp_match_meta_params)
        if is_duplicated:
            self.logger.warning(f'You passed the same model twice in a parallel operation but the original can be found at {new_key}')
            return None
        #No duplicate and also not run in parallel so the lock fails no need to release it
        self.matched_sets[model_key][new_key] = tmp_match_meta_params
         
        return matches, unmatched, G
    
    #Running in parallel and using the lock
    ##Checks if also need to add optimal full
    if new_fullmatch_meta_params is not None:
        is_duplicated, new_full_key = self._check_match_dup(model_key,match_method,new_fullmatch_meta_params)
        if is_duplicated:
            self.logger.warning(f'You passed the same model twice in a parallel operation but the original can be found at {new_key}')
        else:
            full_model_key_val = self.matched_sets[model_key]
            full_model_key_val[new_full_key] = new_fullmatch_meta_params
            self.matched_sets[model_key] = full_model_key_val
            self.matched_sets[model_key][new_full_key] = new_fullmatch_meta_params
    
    is_duplicated, new_key = self._check_match_dup(model_key,match_method,tmp_match_meta_params)
    if is_duplicated:
        self.logger.warning(f'You passed the same model twice in a parallel operation but the original can be found at {new_key}')
        return None
        
    model_key_val = self.matched_sets[model_key]
    model_key_val[new_key] = tmp_match_meta_params
    self.matched_sets[model_key] = model_key_val
    self.lock.release()
   
    return matches, unmatched, G"""
        
        st.code(create_matches,language='python')

    st.header('Greedy/Nearest Neighbor Algorithm')

    st.markdown('#### Examples of code required to produce a matched set using the greedy algorithm are below:')
    
    st.code('''#Exact without replacement on lpsm
my_psm.create_matches('psmodel1',match_method="nearest",manual_caliper_distance=0,replace=False)

#Nearest with replacement caliper 0.1 of std on psm
my_psm.create_matches('psmodel1',match_method="nearest",replace=True,caliper_std = 0.1)
    
#Nearest without replacement caliper 0.1 of std on mahalanobis
my_psm.create_matches('psmodel1',match_method="nearest",distance = "mahalanobis",replace=False,caliper_std = 0.1)''' )


    st.markdown("#### If you are interested to see the main function and the helper functions for the greedy algorithm method of the create_matches method, expand on the links below:")

    with st.expander('##### Main Function: match_nearest_neighbor'):
        st.markdown( '''*This performs the greedy matching algorithm and requires a psm score for at least ordering the minority class to perform the matching. This is true even if not using the ps for matching direcly but it will automatically be generated when creating custom datasets! This will return None if duplicated but otherwise if no matches are found it will return an empty dataframe to add to the meta data so that if it gets run again will be treated like a duplicate*''')
        match_nearest_neighbor = """ def _match_nearest_neighbor(self,model_key, comp_df, treat_df, match_method = "nearest", replace = False, distance = "psm",psm_type = "lpsm",n_matches = 1, caliper_std = -1,nearest_start_method = "random", manual_caliper_distance = None, exact_vars=None, distance_metric_params= None, exact_distance="euclidean", use_psm_other=False,**other_match_params):

        args = locals()
        process_dist_args = {k:v for k,v in args.items() if k not in ['self', 'other_match_params']}
        process_dist_args.update()
        
        ##Checks for and processes exact matching if applicable and sets matching meta params appropriately
      exact_distance_tuple = self._process_distance_metric(**process_dist_args)      
        
        ##Return None if duplicated or if other error preventing matches
        if exact_distance_tuple is None:
            return None
        
        ##Exact only matching so exit function
        if exact_distance_tuple[2]['exact_only']:
            return exact_distance_tuple
        
        ##Unpacks the tuple
        exact_matches_dict, unmatchable, match_meta_params, distance, thresh, dist_match_pool, dist_tr_pool, match_pool, tr_pool, tr_arr, comp_arr, ncol_dist_indices, match_pool_cov, new_key = exact_distance_tuple
         
        
        debugger_params = {k:v for k,v in match_meta_params.items() if k in ['match_method','distance','exact_distance','exact_vars','caliper_std','manual_caliper_distance','replace','n_matches']}
        
        if exact_matches_dict is not None:
        
            ###Continue to match using caliper and/or on psm
            #self.logger.info(f'Acquiring Lock for {model_key} ')
            #self.lock.acquire()
            
            try:
                tree = BallTree(dist_match_pool,metric=distance)
                indices, cdist = tree.query_radius(dist_tr_pool,thresh,return_distance=True)
            except SystemError as e:
                self.logger.warning(f'No matches possible for {model_key} under the matching criteria {debugger_params} due to the following error: {e}')
                return None
            dist_nan_bool = list(map(lambda x: all(np.isnan(x)),cdist))
            if all(dist_nan_bool):
                self.logger.warning(f'All distances were np.nan due to distance metric. No matches possible for {model_key} under the matching criteria {debugger_params}')
                return None
            if any(dist_nan_bool):
                if np.isinf(thresh):
                    self.logger.warning(f'No matches possible for {model_key} under the matching criteria {debugger_params} due to negative distances!')
                    return None
                self.logger.warning(f'Negative distances encountered in {model_key} under the matching criteria {debugger_params}. Use results with caution!')    
            match_pool_ids = np.take(match_pool,0,axis=1)
            func = partial(self._impose_exact_matching, n_matches, exact_matches_dict,match_pool_ids)
            
            match_df = np.array(list(itertools.starmap(func,zip(np.take(tr_pool,0,axis=1),indices,cdist))),dtype=object)
        #No exact matching so skip exact matching and make exact matching dictionary values None for later flag
        else:
            try:
                tree = BallTree(dist_match_pool,metric=distance)
                cdist, indices = tree.query(dist_tr_pool,k = n_matches)
            except SystemError as e:
                self.logger.warning(f'No matches possible for {model_key} under the matching criteria {debugger_params} due to the following error: {e}')
                return None
            dist_nan_bool = list(map(lambda x: all(np.isnan(x)),cdist))
            if all(dist_nan_bool):
                self.logger.warning(f'All distances were np.nan due to distance metric. No matches possible for {model_key} under the matching criteria {debugger_params}')
                return None
            
            if any(dist_nan_bool):
                if np.isinf(thresh):
                    self.logger.warning(f'No matches possible for {model_key} under the matching criteria {debugger_params} due to negative distances!')
                    return None
                self.logger.warning(f'Negative distances encountered in {model_key} uunder the matching criteria {debugger_params}. Use results with caution!')   
            tree = True
            func = partial(self._thresh_dist_check,thresh, match_pool,tree,n_matches)
            match_df = np.array(list(itertools.starmap(func, zip(cdist, indices,tr_pool[:,0]))),dtype=object)
            exact_matches_dict = {i:None for i in tr_arr[:,0]}
        
        unmatchable_results_index, = np.where(match_df[:,3]==False)
        if len(unmatchable_results_index)>0:
            tr_id_nans = np.take(match_df[:,0],unmatchable_results_index)
            #Remove mimssing values
            match_df = np.delete(match_df,unmatchable_results_index,axis=0)
            
            #Remove the boolean column indicating match
            match_df = np.take(match_df,np.arange(3),axis=1)
            
            ##Drop unmatchables from tr_pool
            unmatchable_tr_pool_index = self._np_get_first_matching_indices(tr_pool[:,0],tr_id_nans)
            tr_pool = np.delete(tr_pool, unmatchable_tr_pool_index,axis=0)
            
            ##Create list of unmatchable to potentially append to without replacement results
            unmatchable = unmatchable + list(tr_id_nans)
        #Expand into multiple rows if n_matches>1
        match_length_ind = np.array(list(map(len, match_df[:,1])))
        
        if len(match_length_ind)==0:
            self.logger.warning(f'No matches possible for {model_key} due to caliper under the matching criteria {debugger_params}. Try increasing the caliper to allow for more matches!')
            return None
        
        match_df_long=np.c_[np.repeat(match_df[:,0],match_length_ind),np.concatenate(match_df[:,1]),np.concatenate(match_df[:,2])].astype(np.float_)
        
        ###################This is for without replacement!!!####################
        if not replace:
            #Remove duplicated comp_id
            _, comp_index = np.unique(match_df_long[:,1],return_index=True)
            
            match_df_long = np.take(match_df_long, comp_index, axis = 0)
           
            ######Remove matched treated from tr_pool
            #Need to determine if any people were partially-matched in n_matches>1 scenario
            all_tr_id_bincounts = self._calc_np_bincount(match_df_long[:,0].astype(int))
            len_diff = tr_arr.shape[0] + comp_arr.shape[0] - all_tr_id_bincounts.shape[0]
            
            if len_diff>0:
                all_tr_id_bincounts = np.concatenate([all_tr_id_bincounts,np.zeros(len_diff)])
            
            #All tr appearing n_matches
            complete_matched_tr = np.where(all_tr_id_bincounts==n_matches)[0]
            ##All tr appearing at least once but less than n_matches
            partially_matched_tr_id = np.where((all_tr_id_bincounts>0)&(all_tr_id_bincounts<n_matches))
            match_tr_idx = self._np_get_first_matching_indices(tr_pool[:,0],complete_matched_tr)
            tr_pool = np.delete(tr_pool, match_tr_idx ,axis=0)
            
            #Remove matched comparisons from match pool using numpy
            match_comp_idx = self._np_get_first_matching_indices(match_pool[:,0],match_df_long[:,1])
            match_pool = np.delete(match_pool,match_comp_idx,axis=0)
            match_df = []
            nearest_match_params = {'metric':distance}
            
            if isinstance(distance, MahalanobisDistance):
                match_info = f'with following matching criteria: {debugger_params}'
                nearest_match_params.update({'algorithm':'brute','metric':'mahalanobis','metric_params':{'VI':self._compute_matrix_inv(model_key,match_info, match_pool_cov,m=1e-6)}})
            #self.lock.acquire()    
            for tr_row in tr_pool:
                #Create arrays for passing into BallTree without the match ID number
                dist_match_pool = np.take(match_pool,ncol_dist_indices,axis=1)
                dist_tr_row = np.take(tr_row,ncol_dist_indices).reshape(1,-1)
                ####Determine if number of matching columns greater than one (Note: pools contain the match_id!!)
               
                if match_pool.shape[1] ==2:
                    dist_match_pool = dist_match_pool.reshape(-1,1)
                
                ##Handle partially-matched
                tmp_num_matches = n_matches - all_tr_id_bincounts[int(tr_row[0])]
                
                nn = NearestNeighbors(n_neighbors=int(tmp_num_matches),radius=thresh,**nearest_match_params).fit(dist_match_pool)
                ##Note this returns and array of array for cdist and indices so use [0]
                cdist, indices = nn.radius_neighbors(dist_tr_row)
                if len(indices[0])==0:
                    unmatchable.append(tr_row[0])
                    continue
                
                dist_tuple = self._impose_exact_matching(tmp_num_matches, exact_matches_dict, np.take(match_pool,0,axis=1), tr_row[0], indices[0],cdist[0], nn=True)
                
                ##If nothing survives the exact matching then add tr_id to unmatchable
                if dist_tuple is None:
                    unmatchable.append(tr_row[0])
                    continue
                match_df.append([tr_row[0],dist_tuple[0],dist_tuple[1]])
                ##Drop the currently matched from the match pool
                
                match_pool = np.delete(match_pool,dist_tuple[2],axis=0)
            ##Check if no matches were possible with caliper and if so return empty dataframe
            if len(match_df)==0:
                return pd.DataFrame(match_df, columns = ['treat_id','comp_id','dist','group_id']), np.array(unmatchable), match_meta_params, new_key
            ###Append brute force to ball tree ones###
            #Drop missing values
            match_df = np.array(match_df, dtype=object)
            #Explode the comparison list
            match_length_ind = np.array(list(map(len, match_df[:,1])))
            
            match_df = np.c_[np.repeat(match_df[:,0],match_length_ind),np.concatenate(match_df[:,1]),np.concatenate(match_df[:,2])].astype(np.float_)
            
            final_matches = np.vstack([match_df_long, match_df])
            
            #Sort the final matches in order of treat_id in order to generate group ids
            tr_ids = pd.unique(np.take(final_matches,0,axis=1)).astype(int)
            tr_order = np.argsort(np.take(final_matches,0,axis=1))
            final_matches = np.take(final_matches,tr_order,axis=0)
            
            #Create a frequency table for each tr_id to determine length of group_ids
            all_tr_id_bincounts = self._calc_np_bincount(final_matches[:,0].astype(int))[np.take(tr_ids,np.argsort(tr_ids))]
            
            final_matches = np.c_[final_matches,np.repeat(np.arange(len(tr_ids)), all_tr_id_bincounts)]
        ###Matching with replacement##
        else:
            group_num_arr = np.arange(len(match_length_ind))
            final_matches = np.c_[match_df_long, np.repeat(group_num_arr,match_length_ind)]
           
        return pd.DataFrame(final_matches, columns = ['treat_id','comp_id','dist','group_id']), np.array(unmatchable), match_meta_params, new_key """
        
        st.code(match_nearest_neighbor,language="python")

        
    with st.expander('##### Helper 1: process_distance_metric'):
        st.markdown('''*This helper function process the distance metric. It is possible to use:*
    
    1. Exact matching only
    2. PSM Matching only
    3. Covariate matching only
    4. Covariate matching including psm
    5. PSM Matching + Exact
    6. Covariate matching + Exact
    7. Covariate matching including psm + Exact
    8. PSM matching within caliper and then covariate matching within calipers
    
        ''')
        process_distance_metric = """def _process_distance_metric(self,model_key, comp_df, treat_df, match_method = "nearest", replace = False, distance = "psm",psm_type = "lpsm",n_matches = 1, caliper_std = -1,nearest_start_method = "random", manual_caliper_distance = None, exact_vars=None, distance_metric_params= None, exact_distance="euclidean", use_psm_other=False, full_matching = False, n_matches_min =1, n_matches_max = 2, integral_precision = 5):
            
       args = locals()
       match_meta_params = {k:v for k,v in args.items() if k not in ['self','model_key','comp_df','treat_df']}
       if match_method=="nearest":
           debugger_params = {k:v for k,v in args.items() if k in ['match_method','distance','exact_distance','exact_vars','caliper_std','manual_caliper_distance','replace','n_matches']}
       elif match_method=="optimal":
           debugger_params = {k:v for k,v in args.items() if k in ['match_method','distance','exact_distance','exact_vars','caliper_std','manual_caliper_distance','full_matching','n_matches_min','n_matches_max']}
       ps_var = 'lpsm_score' if psm_type=="lpsm" else 'psm_score'
       ps_var_index = np.where(treat_df.columns.values==ps_var)[0][0]
       
       ##Case where psm should be included as one of the matching variables in other distance metric
       if distance != "psm":
           #Include the correct ps_var
           if use_psm_other:
               bad_ps_var = 'psm_score' if ps_var=='lpsm_score' else 'psm_score'
               match_col_inds = [treat_df.columns.get_loc(col) for col in treat_df.columns if col not in [self.match_id, self.treat_var, self.target_var, bad_ps_var]]
           #Exclude the ps_var
           else:
               match_col_inds = [treat_df.columns.get_loc(col) for col in treat_df.columns if col not in [self.match_id, self.treat_var, self.target_var, "lpsm_score", "psm_score"]]
       #Case where psm is the distance to be used for matching so exclude ps_var
       else:
           match_col_inds = [treat_df.columns.get_loc(col) for col in treat_df.columns if col not in [self.match_id, self.treat_var, self.target_var, "lpsm_score", "psm_score"]]
           
       
       ##Need to extract the column indices of any vars that are exact matching and then separate out the remaining matching variables
       exact_only = False
       if exact_vars is not None:
           exact_col_inds = [treat_df.columns.get_loc(col) for col in exact_vars]
           if len(exact_col_inds)==0:
               exact_vars = None
           
           match_col_inds = list(np.setdiff1d(match_col_inds, exact_col_inds))
           
           if len(match_col_inds)==0:
               exact_only=True
       else:
           exact_col_inds = []
       match_meta_params.update({'exact_only':exact_only})
       ##This is the master arrays with all variables
       tr_arr = treat_df.to_numpy()
       comp_arr = comp_df.to_numpy()
       
       if match_method=="nearest":
           ##Create a list of indexes to order tr_pool for greedy matching
           if nearest_start_method=='random':
               rng = np.random.default_rng(self.random_seed)
               sort_ind = np.arange(len(tr_arr))
               rng.shuffle(sort_ind)
           
           elif nearest_start_method == "min":
               sort_ind = np.argsort(treat_df.values[:,ps_var_index])
               
           elif nearest_start_method=='max':
               sort_ind = np.argsort(treat_df.values[:,ps_var_index])[::-1]
           
           else:
               sort_ind = np.arange(len(tr_arr))
           #Remove parameter keys not used in nearest neighbor
           irrelvant_keys = [match_meta_params.pop(k) for k in ["full_matching","n_matches_min","n_matches_max", "integral_precision"]]
           ##Update the master tr_array to be sorted
           tr_arr = np.take(tr_arr,sort_ind,axis=0)
           
       elif match_method=='optimal':
           irrelvant_keys = [match_meta_params.pop(k) for k in ["nearest_start_method",'replace','n_matches']]
           n_tr = tr_arr.shape[0]
           n_c = comp_arr.shape[0]
           if n_c<n_matches_min*n_tr:
               self.logger.exception(f'Your current dataset contains {n_tr} treated and {n_c} comparison observations. There are not enough comparisons available for your specified minimum of 1:{n_matches_min} matching to make this match set feasible. Either lower the n_matches_min threshold or set the parameter full_match = True!')
               raise ValueError
       ###Determines which kind of distance to use and create the tr_pool and match_pool accordingly
       if distance=='psm':
           tr_pool = np.take(tr_arr,[0, ps_var_index], axis=1)
           match_pool = np.take(comp_arr,[0,ps_var_index],axis=1)
           distance = 'euclidean'
       ##This is any valid distance not with psm 
       else:
           match_meta_params.update({'psm_type':None})
           tr_pool = np.take(tr_arr, [0] + match_col_inds, axis=1)
           match_pool = np.take(comp_arr,[0]+match_col_inds,axis=1)

       #Create arrays for passing into BallTree without the match ID number
       ncol_dist_indices = np.arange(1,match_pool.shape[1])
       dist_match_pool = np.take(match_pool,ncol_dist_indices,axis=1)
       dist_tr_pool = np.take(tr_pool,ncol_dist_indices,axis=1)
       ####Determine if number of matching columns greater than one (Note: pools contain the match_id!!)
       if match_pool.shape[1] ==2:
           if not exact_only:
               if distance=='mahalanobis' or distance=='rank_mahalanobis':
                   distance = 'euclidean'
                   match_meta_params.update({'distance':distance})
                   self.logger.warning(f'Mahalanobis metric requires at least two features but got only 1! Defaulting to euclidean distance for {model_key} under the matching criteria {debugger_params}')
           dist_match_pool = dist_match_pool.reshape(-1,1)
           dist_tr_pool = dist_tr_pool.reshape(-1,1)
       else:
           if distance=='mahalanobis':
               ##First check if possible and if covariance is positive definite
               if not np.all(np.linalg.eigvals(np.cov(dist_match_pool.T))>=0):
                   self.logger.warning(f'No matches possible for {model_key} under the matching criteria {debugger_params} because the covariance matrix is not positive definite!')
                   return None
               ##Possibly use the total covariance?
               #Need to format correctly and calculating np.cov requires columns as rows so transpose!
               tr_mean = dist_tr_pool.mean(axis=0)
               comp_mean = dist_match_pool.mean(axis=0)
               tr_mean_centered = dist_tr_pool.copy()-tr_mean
               comp_mean_centered = dist_match_pool.copy()-comp_mean
               tr_sigma = np.cov(tr_mean_centered.T)
               comp_sigma = np.cov(comp_mean_centered.T)
               match_pool_cov = (tr_sigma+comp_sigma)/2
               match_meta_params.update({'mahal_cov':match_pool_cov})
               try:
                   distance = metrics.DistanceMetric.get_metric('mahalanobis', V = match_pool_cov)
               except LinAlgError as e:
                   self.logger.warning(f'The following LinAlgError arose in calculating the inverse of the covariance matrix for mahalanobis distance of {model_key} under the matching criteria {debugger_params}: "{e.args[0]}". Unable to return matches.')
                   return None
                   
           elif distance=='rank_mahalanobis':

               ##First check if possible and if covariance is positive definite
               if not np.all(np.linalg.eigvals(np.cov(dist_match_pool.T))>=0):
                   self.logger.warning(f'No matches possible for {model_key} under the matching criteria {debugger_params} distance because the covariance matrix is not positive definite!')
                   return None
               #Returns inverse
               match_pool_cov = self._get_rank_mahal_covariance(dist_tr_pool,dist_match_pool)
               match_info = f'with following matching criteria: {debugger_params}'
               inv_covmat = self._compute_matrix_inv(model_key, match_info, match_pool_cov)
               
               match_meta_params.update({'mahal_cov':match_pool_cov})
               try:
                   distance = metrics.DistanceMetric.get_metric('mahalanobis', VI = inv_covmat)
               except LinAlgError as e:
                   self.logger.warning(f'The following LinAlgError arose in calculating the inverse of the covariance matrix for rank_mahalanobis distance of {model_key} under the matching criteria {debugger_params}: "{e.args[0]}". Unable to return matches.')
                   return None
       
       ##Manual caliper is for directly specifying and overrides the caliper_std if not None
       if manual_caliper_distance is None:
           ##Calculate the threshold value for caliper and make it infinite if not specified
           pooled_sd = self._get_pooled_SD(model_key, ps_var)
           if caliper_std>0:
               thresh = caliper_std*pooled_sd
           elif caliper_std==0:
               self.logger.warning(f'This must be a multiple of the pooled standard deviation so an entry of 0 will most likely result in very few matches for {model_key} under the matching criteria {debugger_params}! Use a value of -1 to remove the caliper if this is your intention')
               thresh=0
           else:
               thresh = np.inf
       else:
           match_meta_params.update({'caliper_std':None})
           thresh = manual_caliper_distance
      
       match_meta_params.update({'max_valid_threshold':thresh})
       
       ##Perform exact matching on portion of subsets specified by user
       if exact_vars is not None:
           if len(exact_col_inds)==1:
               if exact_distance=='mahalanobis':
                   exact_distance = 'euclidean'
                   match_meta_params.update({'exact_distance':exact_distance})
                   self.logger.warning(f'Mahalanobis metric requires at least two features but got only one for exact matching of {model_key} under the matching criteria {debugger_params}! Defaulting to euclidean distance for exact matching algorithm')
               exact_tr_pool = np.take(tr_arr, exact_col_inds, axis=1).reshape(-1,1)
               exact_match_pool = np.take(comp_arr,exact_col_inds,axis=1).reshape(-1,1)
           else:
               exact_tr_pool = np.take(tr_arr, exact_col_inds, axis=1)
               exact_match_pool = np.take(comp_arr,exact_col_inds,axis=1)
               
               if exact_distance=='mahalanobis':
                   exact_cov_mat = np.cov(exact_match_pool.T)
                   ##First check if possible and if covariance is positive definite
                   if not np.all(np.linalg.eigvals(exact_cov_mat)>=0):
                       self.logger.warning(f'No matches possible for {model_key} under the matching criteria {debugger_params} because the covariance matrix for exact matching variables is not positive definite!')
                       return None
                   
                   
                   #Need to format correctly and calculating np.cov requires columns as rows so transpose!
                   try:
                       exact_distance = metrics.DistanceMetric.get_metric('mahalanobis', V =exact_cov_mat )
                   except LinAlgError as e:
                       self.logger.exception(f'The following LinAlgError arose in calculating the inverse of the covariance matrix for exact matching mahalanobis distance of {model_key} under the matching criteria {debugger_params}: "{e.args[0]}". Unable to return matches.')
                       raise ValueError
                       
               elif exact_distance=='rank_mahalanobis':
                   
                   exact_match_pool_cov = self._get_rank_mahal_covariance(exact_tr_pool,exact_match_pool)
                   if not np.all(np.linalg.eigvals(exact_match_pool_cov)>=0):
                       self.logger.warning(f'No matches possible for {model_key} under the matching criteria {debugger_params} because the covariance matrix for exact matching variables is not positive definite!')
                       return None
                   
                   match_info = f'with following matching criteria: {debugger_params}'
                   exact_inv_covmat = self._compute_matrix_inv(model_key, match_info, exact_match_pool_cov)
                   try:
                       exact_distance = metrics.DistanceMetric.get_metric('mahalanobis', VI = exact_inv_covmat)
                   except LinAlgError as e:
                       self.logger.exception(f'The following LinAlgError arose in calculating the inverse of the covariance matrix for exact matching rank_mahalanobis distance of {model_key} under the matching criteria {debugger_params}: "{e.args[0]}". Unable to return matches.')
                       raise ValueError
           
           ###This returns exact_matches_dict if further matching is required. Note that match_meta_params['exact_only'] can be used by calling function to determine if final matches can be returned to its own parent function!
           
           if exact_only:
               #Create new meta dictionary based on exact matching parameters to avoid unnecessary duplicates
               match_meta_params = {k:v for k,v in match_meta_params.items() if k in ['match_method','exact_distance','exact_vars','exact_only']}
               match_meta_params.update({'match_method':"exact"})
               
               ##Check for duplicate matches (this is last place where updates occur in meta dict)
               #self.lock.acquire()
               is_duplicated, new_key = self._check_match_dup(model_key,match_method,match_meta_params)
               #self.lock.release()
               if is_duplicated:
                   return None
               
               #Create exact matches dictionary
               #self.lock.acquire()
               try:
                   exact_tree = BallTree(exact_match_pool,metric=exact_distance)
                   exact_matching_inds = exact_tree.query_radius(exact_tr_pool,0)
               except SystemError as e:
                   self.logger.warning(f'No matches possible for {model_key} under the matching criteria {debugger_params} due to the following error: {e}')
                   return None
               exact_matches_dict = {tr_arr[i,0]:np.take(comp_arr[:,0],exact_matching_inds[i],axis=0) for i in range(len(exact_matching_inds))}
               matches, unmatchable = self._exact_only_matching(model_key, tr_arr, comp_arr, exact_matches_dict)
               return matches, unmatchable, match_meta_params, new_key
           
           ##This is for partial exact matching
           else:
               ##Check for duplicate matches (this is last place where updates occur in meta dict)
               is_duplicated, new_key = self._check_match_dup(model_key,match_method,match_meta_params)
               if is_duplicated:
                   return None
               try:
                   exact_tree = BallTree(exact_match_pool,metric=exact_distance)
                   exact_matching_inds = exact_tree.query_radius(exact_tr_pool,0)
                   
               except SystemError as e:
                   self.logger.warning(f'No matches possible for {model_key} under the matching criteria {debugger_params} due to the following error: {e}')
                   return None
               exact_matches_dict = {tr_arr[i,0]:np.take(comp_arr[:,0],exact_matching_inds[i],axis=0) for i in range(len(exact_matching_inds))}
               
       #No exact matching is required but dictionary with None is required in downstream processing
       else:
           ##Check for duplicate matches (this is last place where updates occur in meta dict)
           is_duplicated, new_key = self._check_match_dup(model_key,match_method,match_meta_params)
           if is_duplicated:
               return None
           exact_matches_dict = None
           
       #Further matching is required so matches are replaced with dict containing legal matches and calling function can add onto the empty unmatchable list. This is for consistency so calling function receives 3rd tuple element as the matching meta params as required for the create_matches method. The remaining arguments are necessary for other matching algorithms
       
       if match_meta_params['distance'] not in ['mahalanobis','rank_mahalanobis']:
           match_pool_cov = None
       return exact_matches_dict, [], match_meta_params, distance, thresh, dist_match_pool,dist_tr_pool, match_pool, tr_pool, tr_arr, comp_arr, ncol_dist_indices, match_pool_cov, new_key
"""
        
        st.code(process_distance_metric,language="python")   
    
    
    with st.expander('##### Helper 2: exact_only_matching'):
        st.markdown('''*This function takes a tr_arr and comp_arr because the only way this function is executed is if all variables are used so the structure of each array is that first three columns are match_id, target_var and treat_var respectively and the last two columns are psm_score and lpsm_score *''')
        exact_only_matching = """                   
        def _exact_only_matching(self, model_key, tr_arr, comp_arr, match_dict):
            
    match_length_ind = np.array(list(map(len, match_dict.values())))
    
    unmatchable_results_index, = np.where(match_length_ind==0)
    if len(unmatchable_results_index)>0:
        tr_id_nans = np.take(tr_arr[:,0],unmatchable_results_index)
       
        ##Create list of unmatchable to potentially append to without replacement results
        unmatchable = list(tr_id_nans)
    else:
        unmatchable = []
    
    ##Function to unravel all tr_id and comp_id pairs
    def _unravel_func(tr_id,comp_matches_arr):
        if len(comp_matches_arr)==0:
            return [tr_id,np.nan,False]
        return list(map(lambda comp_id: (tr_id,comp_id,True), comp_matches_arr))
    
    all_pairs = np.vstack(list(itertools.starmap(_unravel_func, match_dict.items())))
    all_matches_idx, = np.where(np.take(all_pairs,2,axis=1))
    all_matches = np.take(np.take(all_pairs,all_matches_idx, axis=0),np.array([0,1]),axis=1)
    
    all_comps = np.take(all_matches,1,axis=1)
    
    duplicate_comp_ids, = np.where(self._calc_np_bincount(all_comps.astype(int))>1)
    
    def _match_tr_to_tr(all_matches, comp_dup):
        ##Find all occurences in all_matches where comp_dup appears and remove first one since it's in master
        all_comps = np.take(all_matches,1,axis=1)
        all_trs = np.take(all_matches,0,axis=1)
        tmp_comp_idx, = np.where(all_comps==comp_dup)
        master_tr_id = np.take(all_trs,tmp_comp_idx[0])
        rows_ids_to_delete = tmp_comp_idx[1:]
        other_tr_ids = np.take(all_trs,rows_ids_to_delete)
        
        tr_tr_matches = np.array(list(map(lambda x: (master_tr_id,x),other_tr_ids)))
        return tr_tr_matches, rows_ids_to_delete
    func_match_tr = partial(_match_tr_to_tr, all_matches)
    tr_match_arr_bunch = list(map(func_match_tr, duplicate_comp_ids))
    ##Unpack the new matches and also the indices to remove
    
    tr_tr_matches = np.vstack(list(map(lambda x: x[0],tr_match_arr_bunch)))
    rows_to_delete = np.concatenate(list(map(lambda x: x[1],tr_match_arr_bunch)))
    all_matches = np.delete(all_matches,rows_to_delete,axis=0)
    
    ##Add the new tr rows but need to filter out duplicates
    _ , unique_idx = np.unique(np.take(tr_tr_matches,1,axis=1),return_index=True)
    tr_tr_matches = np.take(tr_tr_matches,unique_idx, axis=0)
   
    all_matches = np.vstack([all_matches, tr_tr_matches])
    
    #Create a frequency table for each tr_id to determine length of group_ids
    tr_ids = np.unique(np.take(all_matches,0,axis=1)).astype(int)
    all_tr_id_bincounts = self._calc_np_bincount(np.take(all_matches,0,axis=1).astype(int))[tr_ids]
    #Sort the final matches in order of treat_id in order to generate group ids
    tr_order = np.argsort(np.take(all_matches,0,axis=1))
    all_matches = np.take(all_matches,tr_order,axis=0)
    
    return pd.DataFrame(np.c_[all_matches, np.zeros(len(all_matches)), np.repeat(np.arange(len(tr_ids)), all_tr_id_bincounts)],columns = ['treat_id','comp_id','dist','group_id']), unmatchable """
        
        st.code(exact_only_matching,language="python")
        
    with st.expander('##### Helper 3: impose_exact_matching'):
        st.markdown('''*This helper function returns the matching comp_ids and also the mean of their distance. n_matches is the number of desirable matches to return. exact_matches_dict is a dictionary with tr_ids as keys and possible array of comp_ids as values. match_pool_ids is an array of all match_ids for consistent numbering. tr_id is single treatment id with which to find matches. If nn is True that means this function is used in the without replacement manual loop and so must also return the indices of the final matches so they can be dropped from match pool for next iteration*''')
        impose_exact_matching = """def _impose_exact_matching(self, n_matches, exact_matches_dict, match_pool_ids, tr_id, match_arr_inds,dist_arr,nn=False):
            
    matching_comp_ids = np.take(match_pool_ids,match_arr_inds,axis=0)
    
    feasible_matches = exact_matches_dict[tr_id]
    if feasible_matches is not None:
        #finds intersection
        matching_comp_ids = np.intersect1d(feasible_matches, matching_comp_ids)
    
    if len(matching_comp_ids)==0:
        if not nn:
            return tr_id, np.array([np.nan]),np.nan, False
        return None
    #Finds index number of matches within exact guidelines for match_pool - These indices should be in match_arr_inds if no exact matching this should just be the whole match_arr_ind
    matching_match_pool_ids = self._np_get_first_matching_indices(match_pool_ids,matching_comp_ids)
    #Get the index of match_arr_inds in order to get the correct corresponing distances
    dist_arr_index = self._np_get_first_matching_indices(match_arr_inds,matching_match_pool_ids)
    ##Combine the match_arr_inds and dist_arr to 2d array bc need to use argsort on distance and need to return the match_pool index that gets selected along with the actual comp ids Order of columns: comp_ids, comp_arr_ind, dist
    comb_ind_dist_arr = np.c_[matching_comp_ids, np.take(match_arr_inds,dist_arr_index),np.take(dist_arr,dist_arr_index)]
    
    tmp_n_matches = min(n_matches,len(matching_comp_ids))
    
    closest_dist_idx = np.argsort(np.take(comb_ind_dist_arr,2,axis=1))[:int(tmp_n_matches)]
    
    #closest_dist = np.take(dist_arr,closest_dist_idx) 
    comb_ind_dist_arr = np.take(comb_ind_dist_arr,closest_dist_idx,axis=0)
    
    #Get the matches index from the match_pool_ids
    if not nn:
       
        return np.array([tr_id]), np.take(comb_ind_dist_arr,0,axis=1), np.take(comb_ind_dist_arr,2,axis=1), True
    
    return np.take(comb_ind_dist_arr,0,axis=1), np.take(comb_ind_dist_arr,2,axis=1), np.take(comb_ind_dist_arr,1,axis=1).astype(int) """
        
        st.code(impose_exact_matching,language="python")
   
    with st.expander('##### Helper 4: check_match_dup'):
        st.markdown( '''*This function does two main things. It first checks for duplicated matching scenarios and also returns a new key for adding the matched dataset and meta parameters at the end. Also initializes the model key if this is the first match using it* ''')
        check_match_dup = """def _check_match_dup(self, model_key, match_method, meta_params,full_match_check = False):
                       
    ###Remove any array-like object in meta params for easier comparison like df, untreated, mahal_cov etc
    meta_params = {k:v for k,v in meta_params.items() if k not in ['df','unmatched_tr','mahal_cov']}
    
    match_model_dict = self.matched_sets.get(model_key)
    if  match_model_dict is not None:
        if len(match_model_dict)==0:
            #Initialize the model key if for example optimal matching fails and then reruns with full
            self.matched_sets[model_key] = {}
            return (False, match_method + "0")
            
        #Checks if match method has already been run (i.e., maybe different calipers)
        all_existing_match_methods = np.array([re.split(r'\d+$',k)[0] for k in match_model_dict])
        if match_method in all_existing_match_methods:
            ##Grab index of all 
            existing_method_key_ind, = np.where(all_existing_match_methods==match_method)
            all_check_keys = np.take(np.array(list(match_model_dict)),existing_method_key_ind)
            for key in all_check_keys:
                #remove df from params and any other array-like param
                current_meta_params = {k:v for k,v in match_model_dict.get(key).items() if k not in ['df','unmatched_tr','mahal_cov', 'Total_unique_obsvervations', 'Total_unique_treated', 'Total_unique_comparisons', 'weights', 'match_stats_wts', 'match_stats']}
                
                if match_method=='optimal':
                    #Remove the optimal match key because it is unknown until later and the original key is full_matching
                    full_match_ind = current_meta_params.pop('optimal_matching_ratio')
                    if full_match_ind=="full":
                        current_meta_params['full_matching'] = True
                        ##Need to also remove the n_matches_min and n_matches_max keys bc not useful in full_matching
                        meta_params = {k:v for k, v in meta_params.items() if k not in ['n_matches_min','n_matches_max']}
                        current_meta_params = {k:v for k, v in current_meta_params.items() if k not in ['n_matches_min','n_matches_max']}
                        
                    else:
                        
                        current_meta_params['full_matching'] = False
                
                if current_meta_params==meta_params:
                    ##Return the key for easy access to not have to rerun!
                    if full_match_check:
                        return True, key
                    self.logger.warning(f'Matched set with current parameters already exists for this data set and can be found with key {key} within {model_key} for matched_sets attribute!')
                    return (True, None)
           
            return (False, match_method + str(len(existing_method_key_ind)))
    
    #First match using this model
    else:
        #Initialize the model key if this is the first matched set
        self.matched_sets[model_key] = {}
    return (False, match_method + "0") """
        
        st.code(check_match_dup,language="python")
   
    
    st.header('Dynamic/Optimal Matching Algorithms')
    
    st.markdown('#### Examples of code required to produce a matched set using the dynamic algorithm are below:')
    
    st.code('''#Mahalanobis with caliper pair
my_psm.create_matches('psmodel1',match_method="optimal",distance = "mahalanobis",caliper_std = -1, full_matching = False, n_matches_min =1, n_matches_max=1)

#Euclidean with caliper pair
my_psm.create_matches('psmodel1',match_method="optimal",distance = "euclidean",caliper_std=-1, full_matching = False, n_matches_min =1, n_matches_max=1)

#Full matching with mahalanobis
my_psm.create_matches('psmodel1',match_method = "optimal",distance="mahalanobis",full_matching=True)
             ''')
    
    st.markdown("#### If you are interested in the main function and the helper functions for the optimal matching algorithm of the create_matches method, expand on the links below:")
    
    
    with st.expander('##### Main Function: optimal_matching'):
        st.markdown('''
        *This function performs optimal matching based on the method described in Rosenbaum 1991 and allows for pair matching including 1 to n, variable ratio matching with specified min and max number of matches, and full matching. These can all be done with and without caliper and can use or not use the psm score. This uses the networkx package to build the transportation algorithm for minimum cost flow. Also one can use exact matching on certain variables to ensure exact match but this may results in an unsolvable optimization problem.*
        ''')
        optimal_matching = """def _optimal_matching(self, model_key, comp_df, treat_df, match_method = "optimal", distance="psm", psm_type="lpsm",full_matching = False, n_matches_min =1, n_matches_max = 2, caliper_std = -1, integral_precision = 5, manual_caliper_distance = None, exact_vars=None, distance_metric_params= None, exact_distance="euclidean", use_psm_other=False,**other_match_params):
            
       args = locals()
       process_dist_args = {k:v for k,v in args.items() if k not in ['self','other_match_params']}
       
       ##Checks for and processes exact matching if applicable and sets matching meta params appropriately
       exact_distance_tuple = self._process_distance_metric(**process_dist_args)
       ##Return None if duplicated or if other error preventing matches
       if exact_distance_tuple is None:
           return None
       
       ##Exact only matching so exit function
       if exact_distance_tuple[2]['exact_only']:
           return exact_distance_tuple
       
       ##Unpacks the tuple and note unmatchable is empty at this point
       exact_matches_dict, unmatchable, match_meta_params, distance, thresh, dist_match_pool, dist_tr_pool, match_pool, tr_pool, tr_arr, comp_arr, ncol_dist_indices, match_pool_cov, new_key = exact_distance_tuple
       
       
       ##This is necessary for the pairwise distances such mahalanobis
       full_matching_metric = match_meta_params.get('distance')
       if full_matching_metric=='psm':
           full_matching_metric="euclidean"
       
       ###Compute distances for optimal matching and processing edges
       opt_construct_tuple = self._optimal_network_construction(model_key, tr_arr, comp_arr, dist_tr_pool, dist_match_pool, distance, match_meta_params,  n_matches_min, n_matches_max , full_matching, thresh,integral_precision,full_matching_metric, exact_matches_dict)
       ##Occurs bc BallTree failed to return dist_arry
       if opt_construct_tuple is None:
           return None
       G, match_dist_arr,comp_keys,tr_ids, unmatchable, match_meta_params = opt_construct_tuple
       
       debugger_params = {k:v for k,v in match_meta_params.items() if k in ['match_method','distance','exact_distance','exact_vars','caliper_std','manual_caliper_distance','full_matching','n_matches_min','n_matches_max']}
       
       full_to_var_match = False
       matches_dict = None
       ##If the full matching was called for and fails (skip full to var match)
       if full_matching:
           try:
               matches_dict = nx.min_cost_flow(G)
           except NetworkXUnfeasible:
               self.logger.warning(f'No flow satisfied all demands due to many to one treatment IDs for {model_key} under the matching criteria {debugger_params}.')
               return pd.DataFrame([],columns = ['treat_id','comp_id','dist','group_id']), np.array([]), match_meta_params, G, new_key, full_to_var_match, exact_distance_tuple
       
       ##This performs the networkx minimum cost network flow algorithm
       #First set full_to_var_match to False to pass to create matches function
       else:
           try:
               matches_dict = nx.min_cost_flow(G)
           except NetworkXUnfeasible:
               self.logger.warning(f'No flow satisfied all demands due to many to one treatment IDs for {model_key} under the matching criteria {debugger_params}. Retrying by pruning after full matching but including the full match as well!')
               full_to_var_match = True
               #Check first if full matching was already performed
               ##Check for duplicate matches (this is last place where updates occur in meta dict)
               full_match_meta_params = copy.deepcopy(match_meta_params)
               full_match_meta_params.update({'full_matching':True})
               full_match_meta_params.pop('optimal_matching_ratio')
               is_duplicated, old_key = self._check_match_dup(model_key,match_method,full_match_meta_params, full_match_check = True)
               if is_duplicated:
                   old_match_meta = self.matched_sets.get(model_key).get(old_key)
                   old_match_df = old_match_meta.get('df')
                   total_unmatchable = old_match_meta.get('unmatched_tr')
                   
                   #Need to check if full match exists but failed and is empty
                   if len(old_match_df)==0:
                       self.logger.warning(f'Pruning failed because full matching exists in {model_key}_{old_key} but is empty!')
                       return pd.DataFrame([],columns = ['treat_id','comp_id','dist','group_id']), np.array([]), match_meta_params, G, new_key, False, exact_distance_tuple
                   
                   #Returns old full match to use in variable matching or None if full matching was tried but failed
                   return old_match_df, total_unmatchable, match_meta_params, None, new_key, full_to_var_match,  exact_distance_tuple
               
               ####Create network based on full_match = True but keep all other current parameters
               opt_construct_tuple = self._optimal_network_construction(model_key,tr_arr, comp_arr, dist_tr_pool, dist_match_pool, distance, full_match_meta_params,  n_matches_min, n_matches_max , True, thresh,integral_precision,full_matching_metric, exact_matches_dict)
               
               if opt_construct_tuple is None:
                   self.logger.warning('Something weird happened when trying full_to_var for {model_key} under matching criteria: {debugger_params}. Should have failed before reaching this point!')
                   return None
               
               G, match_dist_arr,comp_keys,tr_ids, unmatchable, match_meta_params = opt_construct_tuple
               #Try network flow again on full_match
               try:
                   matches_dict = nx.min_cost_flow(G)
                   
               except NetworkXUnfeasible:
                   debugger_params = {k:v for k,v in match_meta_params.items() if k in ['match_method','distance','exact_distance','exact_vars','caliper_std','manual_caliper_distance','full_matching','n_matches_min','n_matches_max']}
                   self.logger.warning(f'No flow satisfied due to caliper for {model_key} under matching criteria: {debugger_params}!')
                   return pd.DataFrame([],columns = ['treat_id','comp_id','dist','group_id']), np.array([]), match_meta_params, G, new_key, False, exact_distance_tuple
               
               
               ##This happens if full match network does not fail for full_to_var
               final_match_arr, n_matches_min, n_matches_max, unmatchable_opt = self._optimal_process_nx_results(model_key,debugger_params,matches_dict, comp_keys, tr_ids, match_dist_arr,True)
               match_meta_params.update({'n_matches_min':n_matches_min, 'n_matches_max':n_matches_max})
               total_unmatchable = np.concatenate([unmatchable, unmatchable_opt])
               
               return pd.DataFrame(final_match_arr,columns = ['treat_id','comp_id','dist','group_id']), total_unmatchable.astype(int), match_meta_params, G, new_key, full_to_var_match, exact_distance_tuple
               
       ## ###This only happens if output of first networkx does not fail! This process the matching dictionary and converts to array for compatibility with the nearest neighbors function
       final_match_arr, n_matches_min, n_matches_max, unmatchable_opt = self._optimal_process_nx_results(model_key, debugger_params, matches_dict, comp_keys, tr_ids, match_dist_arr,full_matching)
       
       if full_matching:
           match_meta_params.update({'n_matches_min':n_matches_min, 'n_matches_max':n_matches_max})
       
       self.logger.info(f'Here is the datatype for unmatchable: {type(unmatchable)} for model {model_key}')
       self.logger.info(f'Here is the datatype for unmatchable_opt: {type(unmatchable_opt)} for model {model_key}')
       total_unmatchable = np.concatenate([unmatchable, unmatchable_opt])
       return pd.DataFrame(final_match_arr,columns = ['treat_id','comp_id','dist','group_id']), total_unmatchable.astype(int), match_meta_params, G, new_key, full_to_var_match, exact_distance_tuple
"""

        st.code(optimal_matching, language="python")    
    

    with st.expander('##### Helper 1: optimal_network_construction'):
        st.markdown('''*This function acts as a switch to determine the demands and capacities and directions of flow for each type of optimal match scenario*''')
        optimal_network_construction = """def _optimal_network_construction(self, model_key, tr_arr, comp_arr, dist_tr_pool, dist_match_pool,distance, match_meta_params,  n_matches_min=1, n_matches_max = 2,full_matching = False, thresh=2,integral_precision = 5, full_matching_metric = "euclidean",exact_matches_dict = None,full_to_var=False):
    
    debugger_params = {k:v for k,v in match_meta_params.items() if k in ['match_method','distance','exact_distance','exact_vars','caliper_std','manual_caliper_distance','full_matching','n_matches_min','n_matches_max']}
    ###Compute distances for optimal matching and processing edges
    #tr_closest_pairs_dict
    opt_dist_tuple = self._optimal_distances_arr(model_key, debugger_params,tr_arr,comp_arr, dist_tr_pool, dist_match_pool,thresh, distance, integral_precision, full_matching,n_matches_min, full_matching_metric,exact_matches_dict,full_to_var)
    
    
    #This can occur for two reasons 1) BallTree fails to produce distances or 2) full to var match failed. If former full match wouldn't work anyway so no point to try variable! Distances happen before network is assigned
    if opt_dist_tuple is None:
        return None
    match_dist_arr,unique_tr,unique_comp, raw_dist, unmatchable = opt_dist_tuple
    
    ##Remove the full_matching boolean from the meta data because we replace with more meaningful such as constant, variable, and full
    full_key = match_meta_params.pop('full_matching')
    
    if full_matching:
        G = self._optimal_create_full_match(match_dist_arr, unique_tr, unique_comp)
        match_meta_params.update({'optimal_matching_ratio': 'full'})
        
    elif not full_matching and n_matches_min==n_matches_max:
        G = self._optimal_create_constant_match(match_dist_arr, unique_tr, unique_comp, n_matches_min, n_matches_max)
        match_meta_params.update({'optimal_matching_ratio': 'constant'})
    
    elif not full_matching and n_matches_min != n_matches_max:
        G = self._optimal_create_variable_match(match_dist_arr, unique_tr, unique_comp, n_matches_min, n_matches_max)
        match_meta_params.update({'optimal_matching_ratio': 'variable'})
    
    else:
        raise ValueError(f'For {model_key} under matching criteria: {debugger_params}, What happened?')
        
    return G, np.c_[match_dist_arr,raw_dist],unique_comp,unique_tr, unmatchable,  match_meta_params """
        
        st.code(optimal_network_construction,language="python")
        

    with st.expander('##### Helper 2: network scenarios'):
        network_scenarios = """def _optimal_create_constant_match(self, match_dist_arr, unique_tr, unique_comp, n_matches_min, n_matches_max):
            G = nx.DiGraph()
            n_tr = len(unique_tr)
            n_c = len(unique_comp)
            overflow_cost = 0
            overflow_demand = n_c-n_matches_max*n_tr
            matched_demand = -n_matches_min*n_tr
            unmatched_demand = -(n_c-n_matches_min*n_tr)
            
            ##Add source and sink to force flow
            G.add_node("source",demand = -n_c)
            G.add_node("sink",demand = n_c)
            
            ##Add nodes for  comparison
            G.add_nodes_from(unique_comp, demand = 0)
            
            ##Adding edges from source to comp
            G.add_weighted_edges_from([("source",comp_id, 0) for comp_id in unique_comp],capacity=1)
            
            ##Add static nodes
            G.add_node("overflow",demand = overflow_demand)
            G.add_node("matched",demand = matched_demand)
            G.add_node("unmatched",demand = unmatched_demand)
           
            #Add nodes for treatment
            G.add_nodes_from(unique_tr, demand = n_matches_min)
           
            ##Adding edges from comp to tr
            G.add_weighted_edges_from(match_dist_arr,capacity = 1)
            
            ##Add edges between static nodes
            G.add_edge("overflow","unmatched",weight=0,capacity =  np.inf)
            G.add_edge("matched","sink",weight=0,capacity = np.inf)
            G.add_edge("unmatched","sink",weight=0,capacity = np.inf)
        
        
            ##Adding edges from comp to overflow
            G.add_weighted_edges_from([(comp_id, "overflow",overflow_cost) for comp_id in unique_comp],capacity = 1)
            
            ##Adding edged from tr to matched
            G.add_weighted_edges_from([(tr_id, "matched",0) for tr_id in unique_tr],capacity = n_matches_max-1)
            return G

        def _optimal_create_full_match(self, match_dist_arr, unique_tr, unique_comp):
            G = nx.DiGraph()
            n_tr = len(unique_tr)
            n_c = len(unique_comp)
            
            ##Add source and sink to force flow
            G.add_node("source",demand = -(n_tr+n_c))
            G.add_node("sink",demand = n_tr+n_c)
            
            ##Add nodes for  comparison all need to get matched
            G.add_nodes_from(unique_comp, demand = 1)
            #Add nodes for treatment all need to get matched
            G.add_nodes_from(unique_tr, demand = 1)
            
            ##Add node for matched which is a placeholder node necessary to make demands sum to zero.
            G.add_node("matched",demand = -(n_tr+n_c))
            
            ##Adding edges from source to tr
            G.add_weighted_edges_from([("source",tr_id, 0) for tr_id in unique_tr],capacity=n_c-n_tr)
            ##Adding edges from tr to comp bc switched
            G.add_weighted_edges_from(match_dist_arr,capacity = 1)
            
            ###Adding edges from comp to placeholder matched node
            G.add_weighted_edges_from([(comp_id, "matched",0) for comp_id in unique_comp], capacity=np.inf)
            
            ##Adding edge from matched to sink
            G.add_weighted_edges_from([("matched","sink",0)], capacity = n_tr+n_c)
            
            return G

        def _optimal_create_variable_match(self, match_dist_arr, unique_tr, unique_comp, n_matches_min, n_matches_max):
            G = nx.DiGraph()
            n_tr = len(unique_tr)
            n_c = len(unique_comp)
            
            #Need to make it expensive to not match comparison people
            max_weight = np.take(match_dist_arr,2,axis=1).max()
            overflow_cost = max_weight*n_c
            overflow_demand = n_c - n_matches_max*n_tr
            matched_demand = -n_matches_min*n_tr
            unmatched_demand = -overflow_demand
            
            ##Add source and sink to force flow
            G.add_node("source",demand = -n_c)
            G.add_node("sink",demand = n_c)
            
            ##Add nodes for  comparison
            G.add_nodes_from(unique_comp, demand = 0)
            
            ##Adding edges from source to comp
            G.add_weighted_edges_from([("source",comp_id, 0) for comp_id in unique_comp],capacity=1)
            
            ##Add static nodes
            G.add_node("overflow",demand = overflow_demand)
            G.add_node("matched",demand = matched_demand)
            G.add_node("unmatched",demand = unmatched_demand)
           
            #Add nodes for treatment
            G.add_nodes_from(unique_tr, demand = n_matches_min)
           
            ##Adding edges from comp to tr
            G.add_weighted_edges_from(match_dist_arr,capacity = 1)
            
            ##Add edges between static nodes
            G.add_edge("overflow","unmatched",weight=0,capacity =  np.inf)
            G.add_edge("matched","sink",weight=0,capacity = np.inf)
            G.add_edge("unmatched","sink",weight=0,capacity = np.inf)
            ##Adding edges from comp to overflow
            G.add_weighted_edges_from([(comp_id, "overflow",overflow_cost) for comp_id in unique_comp],capacity = 1)
            
            ##Adding edged from tr to matched
            G.add_weighted_edges_from([(tr_id, "matched",0) for tr_id in unique_tr],capacity = n_matches_max-1)
        
            return G """

        st.code(network_scenarios,language="python")



    with st.expander('##### Helper 3: optimal_distances_arr'):
        optimal_distances_arr = """def _optimal_distances_arr(self, model_key, debugger_params,tr_arr, comp_arr,dist_tr_pool, dist_match_pool, thresh, distance, integral_precision=5, full_match=False, n_matches_min=1,full_matching_metric = "euclidean",exact_matches_dict = None,full_to_var=False):
            
       ##Need to have all tr_ids in order to match with dist_tr_pool
       unfiltered_tr_ids = np.take(tr_arr,0,axis=1)
       
       #This distance may be formatted specially as in mahalanobis
       try:
           tree = BallTree(dist_match_pool,metric=distance)
           indices, cdist = tree.query_radius(dist_tr_pool,thresh,return_distance=True)
       
       except SystemError as e:
           self.logger.warning(f'No matches possible for {model_key} under matching criteria: {debugger_params} due to the following error: {e}')
           return None
   
       dist_nan_bool = list(map(lambda x: all(np.isnan(x)),cdist))
       if all(dist_nan_bool):
           self.logger.warning(f'All distances were np.nan due to distance metric for {model_key} under matching criteria: {debugger_params} . No matches possible!')
           return None
       
       if any(dist_nan_bool):
           self.logger.warning(f'Negative distances encountered in calculating distance for {model_key} under matching criteria: {debugger_params} . Use results with caution!')  
       
       func = partial(self._optimal_process_dist, np.take(comp_arr,0,axis=1))
       match_dist_arr = np.concatenate(list(itertools.starmap(func,zip(unfiltered_tr_ids, indices,cdist))))
       dist = np.take(match_dist_arr,2,axis=1)
       unmatched_tr_ind, = np.where(np.isnan(dist))
       num_unmatched_tr = len(unmatched_tr_ind)
       
       ##Unmatched treated here means that no edges or nodes from the id will get drawn in network
       if num_unmatched_tr>0:
           unmatchable = np.take(np.take(match_dist_arr,1,axis=1),unmatched_tr_ind)
           match_dist_arr = np.delete(match_dist_arr,unmatched_tr_ind,axis=0)
           bad_dist_tr_ind = np.concatenate(list(map(lambda x: np.where(unfiltered_tr_ids==x)[0],unmatchable)))
           
           dist_tr_pool = np.delete(dist_tr_pool,bad_dist_tr_ind,axis=0)
           unfiltered_tr_ids = np.delete(unfiltered_tr_ids,bad_dist_tr_ind)
           
       else:
           unmatchable = np.array([])
       
       unique_tr = pd.unique(np.take(match_dist_arr,1,axis=1))
       unique_comp = pd.unique(np.take(match_dist_arr,0,axis=1))
       
       n_tr = len(unique_tr)
       n_c = len(unique_comp)
       ##Need to check if caliper removed too many comparisons to make matching feasible and raises error before running the matching algortihm
       if not full_match and n_matches_min*n_tr>n_c:
           #This means that it is impossible to match remaining observations after pruning to meet min match threshold
           if full_to_var:
               return None
           if n_matches_min==1:
               self.logger.exception(f'Your current caliper of {thresh} resulted in {n_tr} treated and {n_c} comparison observations for {model_key} under matching criteria: {debugger_params} . Your specifed minimum number of matches per treated observation of {n_matches_min} implies there are not enough comparisons available to make this match feasible. Either make your caliper less strict and/or set the parameter full_match = True!')
               raise ValueError
           
           self.logger.exception(f'Your current caliper of {thresh} resulted in {n_tr} treated and {n_c} comparison observations for {model_key} under matching criteria: {debugger_params} . Your specifed minimum number of matches per treated observation of {n_matches_min} implies there are not enough comparisons available to make this match feasible. Either make your caliper less strict or lower your nmatches_min argument or set the parameter full_match = True!')
           raise ValueError
       
       ###Need to filter out the infeasible matches if exact is specified (caliper treated already removed)
       if exact_matches_dict is not None:
           
           ##This returns a list of tuples
           ex_dict_func = partial(self._optimal_exact_matching_dict,exact_matches_dict)
           exact_match_arr = np.vstack(list(map(ex_dict_func,exact_matches_dict.keys())))
           #Identify the unmatchable people based on partial exact and remove from array of feasible matches
           unmatch_ind, = np.where(np.take(exact_match_arr, 1,axis=1)==False)
           exact_unmatched = np.take(np.take(exact_match_arr,0,axis=1),unmatch_ind,axis=0)
           unmatchable = np.concatenate([unmatchable,exact_unmatched])
           exact_match_arr = np.delete(exact_match_arr,unmatch_ind,axis=0)
          
          
           #Convert into dictionary with comp,tr tuple as key so to be matched to the match_dist_arr of all possible combinations
           exact_matches_dict = dict(exact_match_arr)
           ex_filter_func = partial(self._optimal_exact_filter,exact_matches_dict)
           match_dist_arr = np.vstack(list(map(ex_filter_func, match_dist_arr)))
           
           infeasible_exact, = np.where(np.isnan(np.take(match_dist_arr,0,axis=1)))
           if len(infeasible_exact)>0:
              match_dist_arr = np.delete(match_dist_arr,infeasible_exact,axis=0)
              unique_comp = pd.unique(np.take(match_dist_arr,0,axis=1))
              unique_tr_exact = pd.unique(np.take(match_dist_arr,1,axis=1))
              if len(unique_tr_exact)<n_tr:
                  
                  
                  #Get the difference between unique_tr vectors to see which one is missing and hence needs to be added to unmatchable if the exact constraints forced it
                  unmatchable = pd.unique(np.concatenate([unmatchable.astype(int), np.setdiff1d(unique_tr, unique_tr_exact).astype(int)]))
                  unique_tr = unique_tr_exact
                  n_tr = len(unique_tr)
                  
                  bad_dist_tr_ind = np.concatenate(list(map(lambda x: np.where(unfiltered_tr_ids==x)[0],unmatchable)))
                  
                  dist_tr_pool = np.delete(dist_tr_pool,bad_dist_tr_ind,axis=0)
                  
                  unfiltered_tr_ids = np.delete(unfiltered_tr_ids,bad_dist_tr_ind)
              
              if len(unique_comp)<n_c:
                  n_c = len(unique_comp)
                  ##Need to check if exact matching constraint removed too many comparisons to make matching feasible and raises error before running the matching algortihm
                  if not full_match and n_matches_min*n_tr>n_c:
                      if n_matches_min==1:
                          self.logger.exception(f'The exact matching constraint resulted in {n_tr} treated and {n_c} comparison observations for {model_key} under matching criteria: {debugger_params} . Your specifed minimum number of matches per treated observation of {n_matches_min} implies there are not enough comparisons available to make this match feasible. Either relax some of your exact matching constraints and/or make your caliper less strict and/or set the parameter full_match = True!')
                          raise ValueError
                      self.logger.exception(f'Your current caliper of {thresh} resulted in {n_tr} treated and {n_c} comparison observations for {model_key} under matching criteria: {debugger_params} . Your specifed minimum number of matches per treated observation of {n_matches_min} implies there are not enough comparisons available to make this match feasible. Either relax some of your exact matching constraints and/or make your caliper less strict and/or lower your n_matches_min argument or set the parameter full_match = True!')    
                      raise ValueError
       
       dist = match_dist_arr[:,2].copy()
       
       match_dist_arr[:,2] = np.round(dist*10**integral_precision).astype(int)
       
       ##Need to flip tr and comp edges direction to flow from tr to comp
       if full_match:
           match_dist_arr = np.take(match_dist_arr,[1,0,2],axis=1)
      
       return match_dist_arr.astype(int),unique_tr.astype(int),unique_comp.astype(int),dist,unmatchable
"""
        
        st.code(optimal_distances_arr, language="python")


    with st.expander('##### Helper 4: optimal_full_to_var_match'):
        optimal_full_to_var_match = """ def _optimal_full_to_var_match(self, model_key, full_matches_arr, n_matches_min, n_matches_max, exact_distance_tuple, match_meta_params, unmatchable):
             ###Get all the data from exact_matches tuple to pass as needed
             debugger_params = {k:v for k,v in match_meta_params.items() if k in ['match_method','distance','exact_distance','exact_vars','caliper_std','manual_caliper_distance','full_matching','n_matches_min','n_matches_max']}
             exact_matches_dict = exact_distance_tuple[0]
             distance = exact_distance_tuple[3]
             thresh = exact_distance_tuple[4]
             dist_match_pool = exact_distance_tuple[5]
             dist_tr_pool = exact_distance_tuple[6]
             full_matching_metric = match_meta_params.get('distance')
             full_matching = False
             integral_precision = match_meta_params.get('integral_precision')
             
             ##Get the dataframe model to identify treated and comparison
             pval_key = self.desc_stats[model_key].get('pval_key')
             index = self.desc_stats[model_key].get('df_index')
             custom = True if pval_key=='custom' else False
             ps_df = self.get_psm_df(index,custom=custom, manual_pval_key = pval_key)
             comp,treat = [ps_df.loc[ps_df[self.treat_var]==i] for i in [0,1]]
             
             comp_arr = comp.to_numpy()
             tr_arr = treat.to_numpy()
             
             comp_list = np.take(comp_arr, 0, axis=1)
             #Identify treatment IDs in comp ID column
             tr_comp_id = np.setdiff1d(np.take(full_matches_arr, 1,axis=1), comp_list)
             tr_id_loc = np.concatenate(list(map(lambda tr_id: np.where(np.take(full_matches_arr,1,axis=1)==tr_id)[0], tr_comp_id)))
            
             new_unmatchable = pd.unique(np.ravel(np.take(np.take(full_matches_arr,[0,1],axis=1),tr_id_loc,axis=0)))
             
             full_matches_arr = np.delete(full_matches_arr,tr_id_loc, axis=0)
             
             tr_id_list = pd.unique(np.take(full_matches_arr, 0, axis=1))
             
             unmatchable = np.concatenate([unmatchable,np.setdiff1d(new_unmatchable, tr_id_list)])
             
             def _trim_matches(full_matches_arr, n_matches_min, n_matches_max, thresh, tr_id):
                 tr_idx, = np.where(np.take(full_matches_arr,0,axis=1)==tr_id)
                 n = len(tr_idx)
                 #Case where less than min and so then we count the returned ids as unmatchable
                 if n< n_matches_min:
                     return np.array([]), np.array([tr_id]), np.take(full_matches_arr[:,1],tr_idx)
                 #Range that is valid so just return whatever is there
                 elif n>= n_matches_min and n<=n_matches_max:
                     return tr_idx, np.array([]), np.array([])
                 #Case where more than max so much select the closest
                 else:
                     all_matches = np.take(full_matches_arr,tr_idx,axis=0)
                     #sort on distance column which is index 2
                     sort_order_idx = np.argsort(np.take(all_matches, 2,axis=1))
                     final_tr_idx = np.take(tr_idx, sort_order_idx)
                     good_idx = np.take(final_tr_idx,np.arange(n_matches_max))
                     bad_idx = np.take(final_tr_idx, np.arange(n_matches_max, len(final_tr_idx)))
                     tr_to_match = np.array([])
                     comp_to_match = np.take(full_matches_arr[:,1],bad_idx)
                     return good_idx, tr_to_match, comp_to_match
                     
             trim_matches_func = partial(_trim_matches, full_matches_arr, n_matches_min, n_matches_max, thresh)
             idx_select_tuple = np.array(list(map(trim_matches_func, tr_id_list)),dtype=object)
             valid_idx = np.concatenate(np.take(idx_select_tuple,0,axis=1)).astype(int)
             tr_id_to_match = np.concatenate(np.take(idx_select_tuple,1,axis=1)).astype(int)
             comp_id_to_match = np.concatenate(np.take(idx_select_tuple,2,axis=1)).astype(int)
             
             ##Now need to find the location of the id's to be matched again from master array to align with dist arrays to feed to optimal_matching network
             
             if len(tr_id_to_match)>0 and len(comp_id_to_match)>=n_matches_min:
                 tr_dist_idx = self._np_get_first_matching_indices(np.take(tr_arr,0,axis=1), tr_id_to_match)
                 comp_dist_idx = self._np_get_first_matching_indices(np.take(comp_arr,0,axis=1), comp_id_to_match)
                 
                 tr_arr_mod = np.take(tr_arr,tr_dist_idx,axis=0)
                 comp_arr_mod = np.take(comp_arr, comp_dist_idx, axis=0)
                 dist_tr_pool_mod = np.take(dist_tr_pool,tr_dist_idx, axis=0)
                 dist_match_pool_mod = np.take(dist_match_pool, comp_dist_idx, axis=0)
                 
                 ##This needs to be variable matching params, and needs full_matching=False
                 opt_construction_tuple = self._optimal_network_construction(model_key, tr_arr_mod, comp_arr_mod, dist_tr_pool_mod, dist_match_pool_mod, distance, match_meta_params,  n_matches_min, n_matches_max , full_matching, thresh,integral_precision,full_matching_metric, exact_matches_dict,full_to_var=True)
                 
                 ##If the network will be infeasible so no more  matches to add to pruned
                 if opt_construction_tuple is None:
                     final_match_arr = np.empty((0,4))
                     unmatchable_opt = np.array([])
                     unmatchable_opt2 = tr_id_to_match
                     G_mod = None
                 
                 else:
                     
                     G_mod, match_dist_arr,comp_keys,tr_ids, unmatchable_opt, match_meta_params = opt_construction_tuple
                     
                     ###Need to check for duplicates to see if variable/constant matching happened before full_match which would be costly
                     is_duplicated, new_key = self._check_match_dup(model_key,"optimal",match_meta_params)
                     if is_duplicated:
                         self.logger.warning(f'Full to Variable matching for {model_key} is unnecessary since duplicate model exists at {new_key}')
                         return None
                     
                     matches_dict=None
                     try:
                         matches_dict = nx.min_cost_flow(G_mod)
                     except NetworkXUnfeasible:
                         self.logger.warning(f'No flow satisfied all demands on full match to variable matching due to caliper for {model_key} under matching criteria: {debugger_params}!')
                         
                     if matches_dict is not None:
                         debugger_params = {k:v for k,v in match_meta_params.items() if k in ['match_method','distance','exact_distance','exact_vars','caliper_std','manual_caliper_distance','full_matching','n_matches_min','n_matches_max']}
                         final_match_arr, n_matches_min, n_matches_max, unmatchable_opt2 = self._optimal_process_nx_results(model_key, debugger_params, matches_dict, comp_keys, tr_ids, match_dist_arr,full_matching)
                     else:
                         final_match_arr = np.empty((0,4))
                         unmatchable_opt2 = tr_ids
             
             #Case where no possible matches to consider
             else:
                 final_match_arr = np.empty((0,4))
                 unmatchable_opt = np.array([])
                 unmatchable_opt2 = tr_id_to_match
                 G_mod = None
              
             #Consolidate all unmatchable treat ids
             total_unmatchable = np.concatenate([unmatchable, unmatchable_opt, unmatchable_opt2] ,dtype=np.float_)
             
             ##Combine all matches
             final_match_arr = np.vstack([np.take(full_matches_arr,valid_idx,axis=0),final_match_arr])
             #Remove old group id column
             final_match_arr = np.take(final_match_arr, np.arange(3),axis=1)
             
             #Create a frequency table for each tr_id to determine length of group_ids
             tr_ids = np.unique(np.take(final_match_arr,0,axis=1)).astype(int)
             all_tr_id_bincounts = self._calc_np_bincount(final_match_arr[:,0].astype(int))[tr_ids]
             #Sort the final matches in order of treat_id in order to generate group ids
             tr_order = np.argsort(np.take(final_match_arr,0,axis=1))
             final_match_arr = np.take(final_match_arr,tr_order,axis=0)
             
             final_match_arr = np.c_[final_match_arr, np.repeat(np.arange(len(tr_ids)),all_tr_id_bincounts)]
             
             return pd.DataFrame(final_match_arr,columns = ['treat_id','comp_id','dist','group_id']), total_unmatchable.astype(int), match_meta_params, G_mod """
        
        st.code(optimal_full_to_var_match, language="python")

    with st.expander('##### Helper 5: optimal_process_nx_results'):
        st.markdown('''*This helper function returns an array with 4 columns (tr_id, comp_id, dist, group_id). Inputs are a dictionary result of nx algorithm along with an array of comparison to treatment edges and distances. For convenience this also requires an array of unique comparison ids and unique treatment ids which were calculated in the calling functions.* ''')
        optimal_process_nx_results = """def _optimal_process_nx_results(self, model_key, match_info, matches_dict,comp_keys,tr_ids, match_dist_arr,full_match=False, tr_closest_pairs_dict=None):
            #Reshape match_dist_arr into a dictionary with keys as tuples of directed edge (comp_id to tr_id) and values are distances
            
            match_dist_dict = {(int(row[0]),int(row[1])):row[3] for row in match_dist_arr}
            
            #Create inner loop map function
            def _inner_loop(match_dist_dict, full_match, comp_id, tr_id):
                match_pair_dist = match_dist_dict[(comp_id,tr_id)]
                if full_match:
                    return np.array([comp_id,tr_id,match_pair_dist],dtype=np.float_)
                return np.array([tr_id,comp_id,match_pair_dist],dtype=np.float_)
            
            #Create outer loop map function which calls the inner loop function inside
            def _outer_loop(matches_dict,match_dist_dict, full_match, comp_id):
                tr_id_dict = matches_dict[comp_id]
                match_tr_list = [tr_id for tr_id, val in tr_id_dict.items() if val>0 and isinstance(tr_id,np.integer)]
                if len(match_tr_list)>0:
                    func_inner = partial(_inner_loop, match_dist_dict, full_match, comp_id)
                    return list(map(func_inner,match_tr_list))
                else:
                    return [np.array([np.nan,comp_id,np.nan])]
                
            func_outer = partial(_outer_loop,matches_dict,match_dist_dict,full_match)
            
            ##If full_match the treated and comparison switch in the network so need to adjust appropriately
            unmatchable = []
            if full_match:
                matched_pairs = list(map(func_outer,tr_ids))
                #Converts list of arrays into one array
                final_matches = np.vstack(matched_pairs)
               
                #Identifies unmatched treated in full matching since all comps are matched
                unmatched_ids, = np.where(np.isnan(np.take(final_matches,0,axis=1)))
                if len(unmatched_ids)>0:
                    #Get all unmatched tr
                    unmatched_tr = np.take(final_matches[:,1],unmatched_ids)
                    
                    #Match tr to closest tr need to make the tr_id column equal to already matched treated
                    def _add_unmatched_tr(matches_dict, unmatchable, tr_id):
                        
                        if matches_dict.get('source').get(tr_id)==1:
                            #Extract nearest neighbor which will be a key in the nested matches dict also use the distance between that tr_id and the closest comp id even though the match is with the tr_id.
                            comp_neighbor_id = list(matches_dict.get(tr_id).keys())[0]
                            match_arr_comp_ind, = np.where(np.take(final_matches,1,axis=1)==comp_neighbor_id)
                            return np.array([np.take(np.take(final_matches,0,axis=1),match_arr_comp_ind)[0], tr_id, match_dist_dict[(tr_id,comp_neighbor_id)]])
                        else:
                            self.logger.warning(f'Something very wrong occured when trying to match treated to treated for {model_key} under the matching criteria: {match_info}! Here is the info for treated_id {tr_id}: {matches_dict.get(tr_id)}')
                            unmatchable.append(tr_id)
                            return(np.array([np.nan,tr_id,np.nan]))
                        
                           
                    #Create treatment pair arr and also check if pairwise distances failed
                    tr_func = partial(_add_unmatched_tr, matches_dict,unmatchable)
                    tr_pair_arr = np.vstack(list(map(tr_func,unmatched_tr)))
                    
                    problem_tr_inds, = np.where(np.isnan(np.take(tr_pair_arr,1,axis=1)))
                    if len(problem_tr_inds)>0:
                        unmatchable = np.take(np.take(tr_pair_arr,1,axis=1),problem_tr_inds)
                        tr_pair_arr = np.delete(tr_pair_arr,problem_tr_inds,axis=0)
                        
                    #Remove the nans
                    final_matches = np.delete(final_matches,unmatched_ids,axis=0)
                    #Append the matched tr to tr to final matches
                    final_matches = np.vstack([final_matches, tr_pair_arr])
                    #Need to update these bc they could switch to the comp_id column!!
                    tr_ids = pd.unique(np.take(final_matches,0,axis=1)).astype(int)
            
            else:
                matched_pairs = list(map(func_outer,comp_keys))
                #Converts list of arrays into one array
                final_matches = np.vstack(matched_pairs)
                
                #Removes unmatched comp
                unmatched_comp, = np.where(np.isnan(np.take(final_matches,0,axis=1)))
                final_matches = np.delete(final_matches,unmatched_comp,axis=0)
            
            #Sort the final matches in order of treat_id in order to generate group ids
            tr_order = np.argsort(np.take(final_matches,0,axis=1))
            final_matches = np.take(final_matches,tr_order,axis=0)
            
            #Create a frequency table for each tr_id to determine length of group_ids
            all_tr_id_bincounts = self._calc_np_bincount(final_matches[:,0].astype(int))[np.take(tr_ids,np.argsort(tr_ids))]
            
            return np.c_[final_matches,np.repeat(np.arange(len(tr_ids)), all_tr_id_bincounts)], all_tr_id_bincounts.min(), all_tr_id_bincounts.max(), unmatchable """
        
        st.code(optimal_process_nx_results, language="python")

if selected=='Phase 3':
    st.title("Phase 3: Assessing the Quality of Matches")

    st.markdown('#### The same metrics calculated in phase 1 on the pre-matched sample were calculated in phase 3 with the addition of calculating an average within-pair difference</sup><sup class="sup">5</sup>. One unique aspect of these metrics is that the differences are normalized using the same pooled standard deviation metric calculated before matching in Phase 1. This provides an appropriate comparison to the baseline measure</sup><sup class="sup">6</sup>. ', unsafe_allow_html=True)
    
    st.markdown('#### For exact matching, we calculated the weighted difference in means because there were differences between the number of treatment and control units in the matched groups</sup><sup class="sup">12</sup>. These weights were assigned by setting the treatment weights to unity and the controls were assigned a weight proportional to the number of treated units divided by the number of control units within each matched group</sup><sup class="sup">12</sup>.  The sum of the control weights was then scaled to equal the total number of controls in the matched set</sup><sup class="sup">12</sup>.',unsafe_allow_html=True)
    
    st.markdown('#### Using the multiple criteria discussed in phase 1 for calculating the baseline metrics, all matched sets created in phase 2 were compared to the baseline metrics. This is crucial to address the King and Nielson critique of propensity score matching</sup><sup class="sup">13</sup>.  Rubin found that in order for regression adjustments to be accepted, the absolute standardized differences of means should be less than 0.25 and the variance ratios should be between 0.5 and two</sup><sup class="sup">10</sup>. We also considered a stricter threshold of 0.1 for the absolute standardized difference in means for each covariate</sup><sup class="sup">14</sup>. Additionally, we considered the number of unmatched treated units in each matched set since discarding too many undermines the external validity</sup><sup class="sup">15</sup>.',unsafe_allow_html=True)
    
    st.markdown('#### Below is an example of using the PSA class to assess the quality of matches where the first argument is the name of the model obtained from phase 1 and the second is the name of the matching model produced in phase 2: ')
    
    st.code("""my_psm.assess_matches('psmodel1','nearest2') 
my_psm.assess_matches('psmodel1','optimal1')  """,language="python")
   
    st.markdown("#### If you are interested to see the main function and the helper functions for phase 3, expand on the links below:")
    
    with st.expander('##### Main Function: assess_matches'):
        assess_matches = """def assess_matches(self, model_key, match_key, outcome_type = "ATT", weights = True, all_indicators = False, save_df = False, print_results = False, overlap_thresh=0.1, digits = 4, sing_mat_corr = 1e-6,shade = False, **plotly_kwargs):
    args = locals()
    plot_kw_check = self._check_dist_plot_keys(plotly_kwargs)
    ##Get the data set with all covariates
    pval_key = self.desc_stats[model_key].get('pval_key')
    index = self.desc_stats[model_key].get('df_index')
    
    ###Check to see if matches already been assesse
    
    if weights:
        matches_meta = self.matched_sets[model_key][match_key].get('match_stats_wts')
    else:
        matches_meta = self.matched_sets[model_key][match_key].get('match_stats')
    mahal_stat = None
    if matches_meta is None:
        if all_indicators==True:
            all_indicators=False
            self.logger.warning(f'Only direct matching variables can be used for first run to allow direct comparison with the mahalanobis metric. Re-run again with all indicator variables if desired for {model_key}_{match_key}.')
    else:
        if len(matches_meta)>0:
            if not all_indicators:
                self.logger.warning(f'Stats already calculated for {match_key} under {model_key}! If you wish to make a histogram run plot_matches and pass this model key and match key. To print results to excel run print_match_stats with the current model key. Returning the previously run match_stats.')
                return matches_meta
                
            mahal_stat = matches_meta.iloc[-1,1]
            
    #Create column names for resulting stats matrix
    stat_colnames = ["Covariates", "Within_match_diff", "Norm_diff", "Var_ratio",	"Resid_ps_var_ratio", "Log_SD_ratio", "Coverage05_c", "Coverage05_tr", "N_c","N_tr","Mean_c", "Mean_tr","SD_c", "SD_tr","Q025_c",	"Q025_tr",	"Q975_c",	"Q975_tr"]
    
    ##Note final_arr will be ordered by group number and then by treatvar with 1 on top!!
    combine_tuple = self._combine_matches_with_data(model_key, match_key, return_df=False,all_indicators=all_indicators, stats=True)
    #Case where matches are empty
    if len(combine_tuple)==2:
        self.logger.warning(f'Unable to assess matches due to 0 matches for {match_key} under {model_key}')
        return None
    
    final_arr, _ , is_binary, pd_cols = combine_tuple
    
    if final_arr.shape[0]==0:
        self.logger.warning(f'The matches for {match_key} under {model_key} is empty. No stats available for this match!')
        basic_desc_stats = pd.DataFrame([],columns = stat_colnames)
        #self.lock.acquire()
        model_key_val = self.matched_sets[model_key]
        model_key_val[match_key]['match_stats'] = basic_desc_stats
        self.matched_sets[model_key] = model_key_val
        #self.lock.release()
        return basic_desc_stats, None
    
    #Create pandas df for histogram and plotly function with appropriate column names
    hist_df =  pd.DataFrame(final_arr.copy(),columns = pd_cols)
    
    ##Splitting ps_df into a comp and tr set
    comp_arr_orig, tr_arr_orig = self._calc_np_groupby_split(final_arr, 2)
    
    #Sort the comp and tr arr by group number to allow wts to be applied later
    comp_arr_orig = np.take(comp_arr_orig, np.argsort(np.take(comp_arr_orig, -1, axis=1)),axis=0)
    tr_arr_orig = np.take(tr_arr_orig, np.argsort(np.take(tr_arr_orig, -1, axis=1)),axis=0)
    
    ##Get unique counts to update meta dict
    n_uniq_tr = pd.unique(np.take(tr_arr_orig,0,axis=1)).shape[0]
    n_uniq_c = pd.unique(np.take(comp_arr_orig,0,axis=1)).shape[0]
    
    model_key_val = self.matched_sets[model_key]
    model_key_val[match_key].update({'Total_unique_obsvervations': n_uniq_tr+n_uniq_c,
                                     'Total_unique_treated': n_uniq_tr,
                                     'Total_unique_comparisons': n_uniq_c})
    self.matched_sets[model_key] = model_key_val
  
    ##Check for ATT vs ATE
    n_tr_out = None if outcome_type.lower()=='att' else n_uniq_tr
    
    #Drop match id, target var, treat_var from comp_arr and tr_arr
    
    comp_arr = np.take(comp_arr_orig,np.arange(3,comp_arr_orig.shape[1]),axis=1)
    tr_arr = np.take(tr_arr_orig,np.arange(3,tr_arr_orig.shape[1]),axis=1)

    ####For overall_balance_stats df####
    ##Overlap metric
    c_psvals, c_lpsvals = np.take(comp_arr,[-4,-3],axis=1).T
    tr_psvals, tr_lpsvals = np.take(tr_arr,[-4,-3],axis=1).T        

    tr_lp_overlap = self._calc_ps_overlap(tr_lpsvals,c_lpsvals,overlap_thresh)
    tr_p_overlap = self._calc_ps_overlap(tr_psvals,c_psvals,overlap_thresh)
    c_lp_overlap = self._calc_ps_overlap(c_lpsvals,tr_lpsvals,overlap_thresh)
    c_p_overlap = self._calc_ps_overlap(c_psvals,tr_psvals,overlap_thresh)
    
    ##Calculate group means (this includes all even dist and group_id)
    comp_mean = comp_arr.mean(axis=0)
    tr_mean = tr_arr.mean(axis=0)
    
    ##MHD balance metric
    ##Need to remove last 4 which include the ps vars, dist group_id
    stat_type = f"assessing matches for {match_key}"
    covar_only_indices = np.arange(0,(len(comp_mean)-4))
    if mahal_stat is None:
        mhd_bal = self._calc_mhd_balance(model_key, stat_type, np.take(comp_arr,covar_only_indices,axis=1),np.take(tr_arr,covar_only_indices, axis=1),np.take(comp_mean,covar_only_indices),np.take(tr_mean,covar_only_indices),sing_mat_corr)
    else:
        mhd_bal = mahal_stat
    
    ##Calculate group std according to categorical versus continuous
    #Get is_binary from meta data to indicate which column is binary (Need to start at 3 bc match_id, target,and treat var)
    n_cols = hist_df.shape[1]
    is_binary = np.take(is_binary,np.arange(3,n_cols-2))
    
    #This includes the covars, the ps_vars, (no dist or group_id)
    no_dist_grp_id = np.arange(0,len(is_binary))
    model_info = f'during calculation of standard deviation for assessing matches for {match_key}'
    std_func = partial(self._calc_std,model_key, model_info)
    
    comp_sd = np.array(list(itertools.starmap(std_func,zip(np.take(comp_arr,no_dist_grp_id,axis=1).T, is_binary,comp_mean))))
    tr_sd = np.array(list(itertools.starmap(std_func,zip(np.take(tr_arr,no_dist_grp_id,axis=1).T, is_binary,tr_mean))))
    
    #Calculate within matching difference
    model_info = f'during calculation of within match difference for assessing matches for {match_key}'
    wts_covar_ind = np.concatenate([[0],np.arange(3,tr_arr_orig.shape[1])])
    within_pair_diff, comp_wt, tr_wt, comp_grp_ct, tr_wts_all = self._calc_within_match_diff(model_key, model_info,np.take(comp_arr_orig,wts_covar_ind,axis=1), np.take(tr_arr_orig,wts_covar_ind,axis=1), comp_sd, tr_sd, n_uniq_c,n_tr_out, weights)
    #Add the weights to the meta data
  
    model_key_val = self.matched_sets[model_key]
    model_key_val[match_key]['weights']=np.vstack([tr_wt, comp_wt])
    self.matched_sets[model_key] = model_key_val
    
    #Calculate counts
    comp_N = np.array(list(itertools.starmap(self._calc_count, zip(np.take(comp_arr, no_dist_grp_id, axis=1).T, is_binary))))
    tr_N = np.array(list(itertools.starmap(self._calc_count, zip(np.take(tr_arr, no_dist_grp_id, axis=1).T, is_binary))))
    
    #Calculate quantiles for 95% interval
    comp_q025 = np.array(list(map(self._calc_quant025,np.take(comp_arr, no_dist_grp_id, axis=1).T)))
    tr_q025 = np.array(list(map(self._calc_quant025,np.take(tr_arr, no_dist_grp_id, axis=1).T)))
    
    comp_q975 = np.array(list(map(self._calc_quant975,np.take(comp_arr, no_dist_grp_id, axis=1).T)))
    tr_q975 = np.array(list(map(self._calc_quant975,np.take(tr_arr, no_dist_grp_id, axis=1).T)))
    
    #Calculate coverage frequency
    comp_coverage = np.array(list(map(self._calc_coverage_freq,np.take(comp_arr, no_dist_grp_id, axis=1).T)))
    tr_coverage = np.array(list(map(self._calc_coverage_freq,np.take(tr_arr, no_dist_grp_id, axis=1).T)))
    
    ##Normalized Differences
    if mahal_stat is None:
        base_stats = self.desc_stats[model_key]['basic_stats'].copy()
        base_stats = base_stats.set_index('Covariates')
        orig_comp_sd = base_stats.loc[pd_cols[3:-2],'SD_c'].to_numpy()
        orig_tr_sd = base_stats.loc[pd_cols[3:-2],'SD_tr'].to_numpy()
    else:
        orig_comp_sd = comp_sd
        orig_tr_sd = tr_sd
    
    if weights:
        model_info = f'during calculation of weighted normalized difference in means for for assessing matches for {match_key}'
        weights_tup = (tr_wts_all, comp_wt)
        norm_diff_func = partial(self._calc_norm_diff, model_key, model_info,weights_tup)
        norm_diff = np.array(list(itertools.starmap(norm_diff_func, zip(np.take(tr_arr,no_dist_grp_id,axis=1).T, np.take(comp_arr,no_dist_grp_id, axis=1).T, orig_tr_sd, orig_comp_sd))))
        
    else:
        
        model_info = f'during calculation of unweighted normalized difference in means for for assessing matches for {match_key}'
        norm_diff_func = partial(self._calc_norm_diff, model_key, model_info, None)
        norm_diff = np.array(list(itertools.starmap(norm_diff_func, zip(np.take(tr_mean,no_dist_grp_id), np.take(comp_mean,no_dist_grp_id), orig_tr_sd, orig_comp_sd))))
    
    ##Ratio of residual variances
    #Remove the ps columns since those are the covars as well as the dist and group id col
    resid_vars = np.arange(-tr_arr.shape[1],-4)
    resid_bin_vars = np.arange(-is_binary.shape[0],-2)
    
    #Get ratio for ps resids
    stat_type = f"assessing matches for {match_key}"
    tr_ps = np.take(tr_arr,-3,axis=1)
    c_ps = np.take(comp_arr,-3,axis=1)
    resid_func = partial(self._get_ps_resid_var_ratios,model_key, stat_type, tr_ps,c_ps)
    resid_var_ps = np.array(list(itertools.starmap(resid_func, zip(np.take(tr_arr,resid_vars,axis=1).T, np.take(comp_arr,resid_vars,axis=1).T,np.take(is_binary,resid_bin_vars)))))
    
    #Pad the missing values for psm with nan, Remember the distance at the end
    resid_var_ps = np.concatenate([resid_var_ps,np.repeat(np.nan,2)])
    
    ##Ratio of variances
    model_info = f'during calculation of ratio of variances for assessing matches for {match_key}'
    var_ratio_func = partial(self._calc_var_ratio, model_key, model_info)
    var_ratio = np.array(list(itertools.starmap(var_ratio_func, zip(tr_sd, comp_sd))))
    
    ##Ratio of log std
    model_info = f'during calculation of log of the standard deviations for assessing matches for {match_key}'
    log_std_ratio_func = partial(self._calc_log_std_ratio,model_key,model_info)
    log_sd_ratio = np.array(list(itertools.starmap(log_std_ratio_func, zip(tr_sd,comp_sd))))
    
    ##Combine all stats into one dataframe
    basic_desc_stats = pd.concat([pd.Series(hist_df.columns[3:-2]),pd.DataFrame(np.c_[within_pair_diff[:-1], norm_diff, var_ratio,resid_var_ps,log_sd_ratio, comp_coverage, tr_coverage, comp_N, tr_N, np.take(comp_mean,no_dist_grp_id), np.take(tr_mean,no_dist_grp_id), comp_sd, tr_sd, comp_q025, tr_q025, comp_q975, tr_q975])],axis=1)
    basic_desc_stats.columns = stat_colnames
    
    ##Convert the N columns to integer for better report
    
    basic_desc_stats['N_tr'] = basic_desc_stats['N_tr'].astype(object)
    basic_desc_stats['N_c'] = basic_desc_stats['N_c'].astype(object)
    
    basic_desc_stats = basic_desc_stats.sort_values("Within_match_diff",ascending=False).reset_index(drop=True)
    
    basic_desc_stats = pd.concat([basic_desc_stats,pd.DataFrame([['dist',within_pair_diff[-1]],["Multivariate measure", mhd_bal]],columns = ["Covariates","Within_match_diff"])]).reset_index(drop=True)
    
    ##This is the first time run
    if mahal_stat is None:
      
        model_key_val = self.matched_sets[model_key]
        if weights:
            
            model_key_val[match_key]['match_stats_wts' ] = basic_desc_stats
            
        else:
            model_key_val[match_key]['match_stats' ] = basic_desc_stats
            
        self.matched_sets[model_key] = model_key_val
        
    if print_results:
        self.print_match_stats(model_key, match_key, weights, custom_match=basic_desc_stats)
    
    ###Create the dist_plot
    std_multiple = plot_kw_check.get('std_multiple')
    ##Set default bin size to 0.1 of a pooled standard deviation
    if std_multiple is None:
        plot_kw_check['std_multiple'] = 0.1
    
    ###This is where the dist plot happens - needs df that has been formatted with comparison, treatment
    hist_fig = self.plotly_dist_plot(model_key,post_match = hist_df,match_key=match_key, **plot_kw_check)
    
    if hist_fig is None:
        self.logger.warning(f'No variation in the propensity scores for {model_key}_{match_key} due to extreme values')
    
    return basic_desc_stats, hist_fig """
        
        
        st.code(assess_matches,language="python")

    with st.expander('##### Helper 1: combine_matches_with_data'):
        combine_matches_with_data = """def _combine_matches_with_data(self, model_key, match_key,return_df = True, extra_vars=None,weights=False, all_indicators=False, stats=False):
    #Get matches
    match_df =  self.matched_sets[model_key][match_key]['df'].to_numpy()
    
    #Check if empty
    if match_df.shape[0]==0:
        if return_df:
            return self.matched_sets[model_key][match_key]['df']
        return self.matched_sets[model_key][match_key]['df'], extra_vars
    
    ##Get the data set with all covariates
    pval_key = self.desc_stats[model_key].get('pval_key')
    index = self.desc_stats[model_key].get('df_index')
    custom = True if pval_key=='custom' else False
    
    if extra_vars is None:
        extra_vars = []
    
    ps_df = self.get_psm_df(index,custom=custom, manual_pval_key = pval_key, extra_vars=extra_vars, all_indicators=all_indicators, stats=stats)
    
    ##If stats is True then get_psm_df returns a tuple so unpack it!
    if stats:
        ps_df, is_binary = ps_df
    
    ##Store column names for later conversion to pandas df for final output
    ps_key_varlist = ps_df.columns.to_list()
    pd_cols = ps_key_varlist + ['dist','group_id']
    
    ###Check for Mahalanobis Distance variation in both exact distance and regualar distance meta data
   
    if self.matched_sets[model_key][match_key]['exact_only']:
        if stats and self.matched_sets[model_key][match_key]['exact_distance'] in ['mahalanobis', 'rank_mahalanobis']:
            mahal_df = self._process_rank_mahal_df(ps_df,self.matched_sets[model_key][match_key]['use_psm_other'])
            match_df = self._calc_dist_after_matching(model_key,match_key,mahal_df)
        else:
            match_df = self._calc_dist_after_matching(model_key,match_key,ps_df)
    else:
        
        if stats and self.matched_sets[model_key][match_key]['distance'] in ['mahalanobis', 'rank_mahalanobis']:
            mahal_df = self._process_rank_mahal_df(ps_df,self.matched_sets[model_key][match_key]['use_psm_other'])
            match_df = self._calc_dist_after_matching(model_key,match_key,mahal_df)
            
        else:
            match_df = self._calc_dist_after_matching(model_key,match_key,ps_df)
    
    
    #Get unique IDs from treatment and comparisons (drop the ones with same match id and group_id)
    match_df_sort = np.take(match_df,np.lexsort((match_df[:,0],match_df[:,-1])),axis=0)
    match_df = match_df_sort[np.concatenate(([True], np.any(match_df_sort[1:] != match_df_sort[:-1],axis=1)))]
    
    ps_df = ps_df.to_numpy()
    
    #Match uniqueIDs from matches to covar matchIDs
    cov_df_index = self._np_get_first_matching_indices(ps_df[:,0],match_df[:,0])
    ps_df =np.take(ps_df,cov_df_index,axis=0)
   
    #Concatenate arrays 
    final_arr = np.hstack([ps_df,np.take(match_df,[1,2],axis=1)])
    
    if return_df:
        #Convert array to pandas dataframe
        final_df = pd.DataFrame(final_arr,columns = pd_cols)
        if weights:
            #Should be a column array (may add other weigts later like ATE)
            weights = pd.DataFrame(self.matched_sets[model_key][match_key].get('weights'),columns=[self.match_id,'group_id','ATT_wts'])
            
            ##Need to sort by treat var and then group number so they match
            #final_df = final_df.sort_values([self.treat_var,"group_id"], ascending=[False, True],ignore_index=True)
            #return pd.concat([final_df,weights],axis=1)
            return pd.merge(final_df,weights, on=[self.match_id,"group_id"])
        
        return final_df
    
    #Othewise this tuple gets passed into assess matches for downstream processing
    return final_arr, extra_vars, is_binary, pd_cols"""


        st.code(combine_matches_with_data,language="python")
        
        
    with st.expander('##### Helper 2: calc_within_match_diff'):
        st.markdown( '''*This takes as input a treatment array and comparison array and returns the normalized difference. If treatment array and/or comparison array have more than one match it will average them before finding difference. Arrays must have covariates for columns and rows number of individuals. Based on pg 355-356 of book. Normalized by sample standard deviations. However, this function also calculates balance statistics for full matching and 1: many matching. ntreat determines if ATE is desired.*''')
        
        calc_within_match_diff = """ def _calc_within_match_diff(self,model_key, model_info, comp_arr, tr_arr, comp_sd, tr_sd,ncomp, ntreat=None, weighted=False):
        ##Note that comp_arr and tr_arr have all the original columns except no target and treat var - index 0 in both is the match_id number!
        #Calculate group means and remove the dist temporarily (-2) and group_id col (-1) This mostly affect many to one matching
        covar_only_indices = np.arange(1,tr_arr.shape[1]-2)
        #Extract mean pairs - This will sort by group_id (last column) 
        pairs_arr_c_2 = self._calc_np_groupby_split(comp_arr, -1)
        pairs_arr_tr_2 = self._calc_np_groupby_split(tr_arr, -1)
        
        #Extract the comp_ids to match to weights at end
        comp_ids = np.concatenate(list(map(lambda x: np.take(x,[0,-1],axis=1),pairs_arr_c_2)))
        tr_ids_all = np.concatenate(list(map(lambda x: np.take(x,[0,-1],axis=1),pairs_arr_tr_2)))
        tr_ids = np.vstack(pd.unique(tuple(zip(*tr_ids_all.T))))
        
        #Calculate the number of tr and comp per group
        n_tr = np.hstack(list(map(lambda x: x.shape[0], pairs_arr_tr_2)))
        n_c = np.hstack(list(map(lambda x: x.shape[0], pairs_arr_c_2)))
        raw_weights = n_tr/n_c
        
        #Calculate group means
        pairs_arr_c =np.vstack(list(map(lambda x: x.mean(axis=0),pairs_arr_c_2)))
        pairs_arr_tr = np.vstack(list(map(lambda x: x.mean(axis=0),pairs_arr_tr_2)))
        
        ##Extract dist because it is already distance and does not need to be subtracted!
        dist = np.take(pairs_arr_tr,-2,axis=1)
        
        #Calculate absolute mean difference and also remove grouping column and temporarily dist
        d_abs = np.abs(np.take(pairs_arr_tr,covar_only_indices,axis=1)-np.take(pairs_arr_c,covar_only_indices,axis=1))
        
        ####Add the distance back onto d_abs
        d_abs = np.c_[d_abs,dist]
        
        #Check to see if variable matching occured
        if np.all(np.isclose(raw_weights,raw_weights[0])):
            avg_dist = np.sqrt(np.sum(d_abs**2,axis=0)/len(n_tr))
            final_weights = np.repeat(raw_weights, n_c)
        
        #Full match weighs or variable matching weights    
        else:
            ##If 1 treated per group normalize to number of treated
            if np.all(n_tr==1):
                
                all_raw_weights_with_group_id = np.c_[np.repeat(raw_weights,n_c),np.repeat(np.arange(len(raw_weights)),n_c)]
                all_raw_weights = np.take(all_raw_weights_with_group_id,0,axis=1)
               
                norm_weights = all_raw_weights
                
                norm_grp_wts = np.array(list(map(lambda x: np.take(x,0,axis=1)[0], self._calc_np_groupby_split(all_raw_weights_with_group_id, 1))))
                
            #Per Green & Stuart 2014, need to normalize to total number of comparisons (ATT weights)
            else:
                all_raw_weights_with_group_id = np.c_[np.repeat(raw_weights,n_c),np.repeat(np.arange(len(raw_weights)),n_c), ]
                all_raw_weights = np.take(all_raw_weights_with_group_id,0,axis=1)
                
                norm_weights = (all_raw_weights*(ncomp/all_raw_weights.sum()))
                norm_grp_wts = np.array(list(map(lambda x: np.take(x,0,axis=1)[0], self._calc_np_groupby_split(all_raw_weights_with_group_id, 1))))
                
            #ATE weights
            if ntreat is not None:
                norm_weights = raw_weights*((ncomp+ntreat)/np.sum(raw_weights))
            
            if weighted:
                w = norm_grp_wts
                wm_func = partial(self._weighted_mean,w)
                avg_dist = np.array(list(map(wm_func, d_abs.T)))
            else:
                
               avg_dist = d_abs.mean(axis=0)

            final_weights = norm_weights
        ##Process the standard deviations (dist will not be included!)
        sd_sq_metric = np.sqrt(comp_sd**2+tr_sd**2)
        
        def _avg_dist_func(model_key, model_info, avg_dist,sd_sq_metric):
            result=np.nan
            try:
                result = avg_dist/sd_sq_metric
            except RuntimeWarning as e:
                self.logger.warning(f'The model {model_key} {model_info} encountered the following error: {e}')
            finally:
                return result
        part_avg_dist_func = partial(_avg_dist_func, model_key, model_info)
        avg_dist_covars = np.array(list(itertools.starmap(part_avg_dist_func, zip(np.take(avg_dist,np.arange(len(avg_dist)-1)),sd_sq_metric))))
        
        #Append the dist metric to final distances
        return np.append(avg_dist_covars,avg_dist[-1]), np.c_[comp_ids,final_weights], np.c_[tr_ids,np.ones(len(tr_ids))],n_c,np.c_[tr_ids_all,np.ones(len(tr_ids_all))] """

        st.code(calc_within_match_diff,language="python")
        
        
    with st.expander('##### Helper 3: calc_dist_after_matching'):
        st.markdown(  '''*This function processes the distance measures after matching in cases with many to one or many to many matches as in full matching* ''')
        calc_dist_after_matching = """def _calc_dist_after_matching(self,model_key, match_key, ps_df):
            
            matches_meta = self.matched_sets[model_key][match_key]
            #Group IDs by group_id regardless of treatment
            match_df = matches_meta.get('df').to_numpy()
            #Melt matches by group number
            match_df_id_only = np.vstack([np.take(match_df,[0,3],axis=1),np.take(match_df,[1,3],axis=1)])
            split_arr = self._calc_np_groupby_split(match_df_id_only, 1)
            grp_len_arr = list(map(len,split_arr))
            
            unique_match_id_list = np.array(list(map(lambda x: pd.unique(np.take(x,0,axis=1)), split_arr)),dtype=object)
            grp_id_list = np.array(list(map(lambda x: np.take(x,1,axis=1)[0],split_arr)))
            
            ##Get the correct distance matching columns from psm_df
            distance_type = matches_meta.get('distance')
            if distance_type=='psm':
                psm_type = "lpsm_score" if matches_meta.get('psm_type')=='lpsm' else "psm_score"
               
                var_dist_arr = ps_df[[self.match_id,psm_type]].to_numpy()
                distance_type = 'euclidean'
            
            else:
                all_match_vars = literal_eval(self.get_model_names().get(model_key))
                exact_vars = matches_meta.get('exact_vars')
                dist_match_vars = list(np.setdiff1d(all_match_vars,exact_vars))
                
                #Case of exact only and distances are all 0 anyway
                if len(dist_match_vars)==0:
                    return np.vstack([np.take(match_df,[0,2,3],axis=1),np.take(match_df,[1,2,3],axis=1)])
                #Case where at least one var is used
                else: 
                    var_dist_arr  = ps_df[[self.match_id] + dist_match_vars].to_numpy()
            
            ncols = var_dist_arr.shape[1]
            ###Create function to map to each list of unique ids and return the avg distance in that group
            def _avg_dist(var_dist_arr,distance_type, match_id_arr,grp_id):
             
                sub_arr_idx = self._np_get_first_matching_indices(np.take(var_dist_arr,0,axis=1), match_id_arr)
                
                full_sub_arr = np.take(var_dist_arr, sub_arr_idx, axis=0)
                sub_arr =np.take(full_sub_arr,np.arange(1,ncols),axis=1)
                if sub_arr.shape[1]==1:
                    sub_arr = sub_arr.reshape(-1,1)
                    if distance_type in ['mahalanobis','rank_mahalanobis']:
                        distance_type = 'euclidean'
                    
                if distance_type in ['mahalanobis','rank_mahalanobis']:
                    
                    match_info = f"in {match_key}"
                    dist_cov = matches_meta.get('mahal_cov')
                    dist_cov_inv = self._compute_matrix_inv(model_key, match_info, dist_cov,m=1e-6)
                    return scispat.distance.pdist(sub_arr,metric='mahalanobis', VI = dist_cov_inv).mean()
                    
                return scispat.distance.pdist(sub_arr,metric=distance_type).mean()   
            
            func_avg_dist = partial(_avg_dist, var_dist_arr,distance_type)
            
            ##Should be one number per group
            avg_dist_arr = np.vstack(list(itertools.starmap(func_avg_dist, zip(unique_match_id_list,grp_id_list))))
            
            #Need to explode to match dimensions of the melted match_df
            exploded_avg_dist = np.repeat(avg_dist_arr, grp_len_arr)
            final_match_arr = np.c_[np.vstack(split_arr),exploded_avg_dist]
            #Reorder columns for consistency
            return np.take(final_match_arr, [0,2,1],axis=1) """

        st.code(calc_dist_after_matching,language="python")
        
    with st.expander('##### Helper 4: Calculating Metrics'):
        calculating_metrics = """@staticmethod
def _calc_count(x, is_binary):
    if is_binary:
        return np.sum(x)
    else:
        return len(x)
@staticmethod
def _calc_coverage_freq(x):
    q975 = np.quantile(x,0.975)
    q025 = np.quantile(x,0.025)
    
    F975 = np.mean(x<=q975)
    F025 = np.mean(x<=q025)
    
    return((1-F975) + F025) 
def _calc_linear_psm(self, model_key, model_info, ps_arr):
    lpsm = np.nan
    try:
        lpsm = np.log(ps_arr/(1-ps_arr))
    except RuntimeWarning as e:
        self.logger.warning(f'The model {model_key} {model_info} encountered the following error: {e}')
    finally:
        return lpsm
    
def _calc_log_std_ratio(self, model_key, model_info, std_t, std_c):
    lstdr = np.nan
    try:
        lstdr = np.log(std_t/std_c)
    except RuntimeWarning as e:
        self.logger.warning(f'The model {model_key} {model_info} encountered the following error: {e} ')
    finally:
        return lstdr

def _calc_mhd_balance(self, model_key, stat_type, X_comp, X_treat, means_comp, means_tr,m = 1e-6):
     '''Calculate the mahalanobis type multivariate balance measure based on Imbens & Rubin 2015, p.314 '''
     dev_c = X_comp - means_comp
     dev_tr = X_treat-means_tr
     
     sigma_tr = dev_tr.T.dot(dev_tr)/(X_treat.shape[0]-1)
     sigma_c = dev_c.T.dot(dev_c)/(X_comp.shape[0]-1)
     sigma_pooled = (sigma_tr + sigma_c)/2
     try:
         sig_inv = np.linalg.inv(sigma_pooled)
     except Exception as e:
         if str(e)=="Singular matrix":
             self.logger.warning(f'Singular matrix detected for MHD balance metric in {model_key} for {stat_type}!')
             sig_inv = np.linalg.inv(sigma_pooled + np.eye(sigma_pooled.shape[1])*m)
             #return sigma_pooled
         else:
             self.logger.info(f'In {model_key} for {stat_type}. What the F happened??????', e)
     return (np.sqrt((means_tr-means_comp).dot(sig_inv).dot((means_tr-means_comp).T)))

def _calc_norm_diff(self,model_key, model_info, weights_tup=None, mean_t=None, mean_c=None, std_t = None, std_c=None):
    '''Calculates the normalized difference in means both weighted and unweighted. For weighted the function requires the the raw data and not just the means. '''
    if weights_tup is None:
        norm_diff = np.nan
        try:
            norm_diff = (np.abs(mean_t - mean_c))/(np.sqrt((std_t**2 + std_c**2)/2))
        except RuntimeWarning() as e:
            self.logger.warning(f'The model {model_key} {model_info} encountered the following error: {e}')
        finally:
            return norm_diff
   
    #Calc weighted tr mean
    w = np.take(weights_tup[0],-1,axis=1)
    tr_wt_mean = self._weighted_mean(w, mean_t)
    #Calc weighted comp mean
    w = np.take(weights_tup[1],-1,axis=1)
    comp_wt_mean = self._weighted_mean(w, mean_c)
    norm_diff = np.nan
    try:
        norm_diff = (np.abs(tr_wt_mean - comp_wt_mean))/(np.sqrt((std_t**2 + std_c**2)/2))
    
    except RuntimeWarning() as e:
        self.logger.warning(f'The model {model_key} {model_info} encountered the following error: {e}')
    finally:
        return norm_diff
    
def _calc_np_bincount(self,arr):
    '''Takes numpy array and applies the numba bincount. Counts gets modified in place and is then returned '''
    M = np.max(arr)
    counts = np.zeros(M + 1, dtype=int)
    self._calc_numba_bincount(arr, counts, len(arr))
    return counts

def _calc_np_groupby_split(self, a, groupcol):  
    '''Replace np.bincount by numba_bincount'''
    
    a = a[a[:, groupcol].argsort()]
    bins = self._calc_np_bincount(a[:,groupcol].astype(int))
    nonzero_bins_idx, = np.where(bins != 0)
    nonzero_bins = np.take(bins, nonzero_bins_idx)
    split_idx = np.cumsum(nonzero_bins[:-1])
    return np.array_split(a, split_idx,axis=0)

@staticmethod
@numba.njit
def _calc_numba_bincount(a, counts, m):
    '''This helper function speeds up the for loop for bincounts '''
    for i in range(m):
        counts[a[i]] += 1

@staticmethod
def _calc_ps_overlap(arr1, arr2,threshold):
    overlap_indicator = [any(np.abs(arr2-x)<threshold) for x in arr1]
    return np.mean(overlap_indicator)

@staticmethod
def _calc_quant025(x):
    return(np.quantile(x,0.025))

@staticmethod
def _calc_quant975(x):
    return(np.quantile(x,0.975))

def _calc_std(self, model_key, model_info, x,is_binary,col_mean):
    '''This function calculates the std based on binary or continuous variable'''    
    
    if is_binary:
        try:
            std = np.sqrt(col_mean*(1-col_mean))
        except RuntimeWarning() as e:
            self.logger.warning(f'The model {model_key} {model_info} encountered the following error: {e}')
        finally:
            return std
    else:
        try:
            std=np.std(x,ddof=1)
        except RuntimeWarning as e:
            self.logger.warning(f'The model {model_key} {model_info} encountered the following error: {e}')
        finally:
            return std

def _calc_var_ratio(self, model_key, model_info, std_t, std_c):
    
    var_rat = np.nan
    try:
        var_rat = std_t**2/std_c**2
    except RuntimeWarning as e:
        self.logger.warning(f'The model {model_key} {model_info} encountered the following error: {e} ')
    finally:
        return var_rat"""
        
        st.code(calculating_metrics,language="python")
        
    st.markdown('#### Below is an example of the code run after assessing matches that prints the results of all matching scenarios of a specific model created in phase 1 to an Excel spreadsheet for easier comparison: ')
    
    st.code('my_psm.print_match_stats("psmodel1")',language="python")
    
    st.markdown('#### Expand to see the function for printing all the match stats: ')
    
    with st.expander('##### Main Function: print_match_stats'):
        print_match_stats = """ def print_match_stats(self,model_key, match_key="all", weights=True, custom_match= None, digits = 4, match_meta_add_on = True, treat_color = "#FF7F0E",comp_color = "#1F77B4", before_color = "GreenYellow",after_color="Orchid", var_boundary_color = 'blue',midline = "red",height = 1500, width=1200):
    args = locals()
    self.logger = self._setup_logger()
    plot_args = {k:v for k,v in args.items() if k in ["comp_color", "before_color", "treat_color", "after_color", "var_boundary_color", 'midline','height','width']}
    
    wts_str = "_wts" if weights else ""
    if custom_match is None:
        ##Extract matched sets
        all_matches_dict = self.get_match_stats(model_key, match_key)
        self.logger.info("Here is the all_matches_dict: ", all_matches_dict)
            
    else:
        all_matches_dict = {model_key:{match_key:custom_match}}
        plot_args.update({'custom_match':custom_match})
        
    ##Extract prematch set for each valid model key
    for mod_key in all_matches_dict:
        #Create new folder if first run
        if not Path(self.main_path + '/' + mod_key).is_dir():
            Path(self.main_path + '/' + mod_key).mkdir(mode=0o700,parents=True,exist_ok=False)
    current_utc_time = self._generate_time_stamp()
    all_matches_dict[mod_key].update({'aaBefore':self.desc_stats[mod_key]['basic_stats']})
    top_keys = list(all_matches_dict.keys())
    
    ##Check if multiple model_keys for different processing
    if len(all_matches_dict)>1:
        ##Case where the first sheet is all models so the names must be altered with model key and match key
        #This will have lots of big tables in each sheet
        
        pass
    elif len(all_matches_dict)==1:
        #Case with one model (each other sheet contains full stats and first sheet is comparison)
        match_comparison = {'Match Comparison':{}}
        titles = {'Match Comparison':{}}
        for mod_key,match_dict in all_matches_dict.items():
            for mat_key, df in match_dict.items():
                
                if mat_key=="aaBefore":
                    match_comparison['Match Comparison'].update({"Before Matching"+" "+mod_key: df[['Covariates',"Norm_diff","Resid_ps_var_ratio","Log_SD_ratio","Coverage05_c","Coverage05_tr"]]})
                    match_comparison.update({"Before Matching"+" "+mod_key:{"Before Matching"+" "+mod_key:df}})
                    titles.update({"Before Matching"+" "+mod_key:{"Before Matching"+" "+mod_key:"Before Matching"+" "+mod_key}})
                    titles['Match Comparison'].update({"Before Matching"+" "+mod_key:"Before Matching"+" "+mod_key})
                    
                else:
                    
                    self.logger.info(f"Here is the current Match: {mod_key}_{mat_key}")
                    ##Print diagnostic plots
                    cat_plot, mean_plot, var_plot = self.plotly_covar_balance(mod_key, mat_key, True, True, None, weights, **plot_args)
                    if cat_plot is not None:
                        cat_plot.write_image(f'{self.main_path}/{mod_key}/{mat_key}_catplot{wts_str}{current_utc_time["file_time"]}.pdf',format='pdf', height = height, width=width)
                    
                    mean_plot.write_image(f'{self.main_path}/{mod_key}/{mat_key}_meanplot{wts_str}{current_utc_time["file_time"]}.pdf',format='pdf',height = height, width=width)
                    
                    var_plot.write_image(f'{self.main_path}/{mod_key}/{mat_key}_varplot{wts_str}{current_utc_time["file_time"]}.pdf',format='pdf',height = height, width=width)
                    
                    match_comparison['Match Comparison'].update({mod_key+'_'+mat_key: df[['Covariates',"Within_match_diff","Norm_diff","Resid_ps_var_ratio","Log_SD_ratio","Coverage05_c","Coverage05_tr"]]})
                    match_comparison.update({mod_key+'_'+mat_key:{mod_key+'_'+mat_key:df}})
                    titles.update({mod_key+'_'+mat_key:{mod_key+'_'+mat_key:mod_key+'_'+mat_key}})
                    titles['Match Comparison'].update({mod_key+'_'+mat_key:mod_key+'_'+mat_key})
        
        titles['Match Comparison'] = {k:titles['Match Comparison'][k] for k in ['Before Matching'+" "+mod_key]+ list(np.setdiff1d(list(titles['Match Comparison'].keys()),['Before Matching'+" "+mod_key]))}
        match_comparison['Match Comparison'] = {k:match_comparison['Match Comparison'][k] for k in ['Before Matching'+" "+mod_key]+ list(np.setdiff1d(list(match_comparison['Match Comparison'].keys()),['Before Matching'+" "+mod_key]))}
        
    else:
        #Case where no valid dict (i.e., no matched sets so user needs to run assess_matched_sets)
        self.logger.warning('No statistics for matching found! Run assess_matches to create statistics and then rerun the print_match_stats_method.')
    
    self._print_excel_dict(match_comparison, f'{self.main_path}/{model_key}/match_results{wts_str}{current_utc_time["file_time"]}.xlsx', table_titles = titles, digits = digits, match_meta_add_on=match_meta_add_on)
    
    return None"""
        
        st.code(print_match_stats,language="python")
        
    with st.expander('##### Helper Function: plotly_covar_balance'):
        plotly_covar_balance = """def plotly_covar_balance(self, model_key, match_key=None, post_match=False, print_model_info=True, custom_match=None, weights=True, treat_color = "#FF7F0E",comp_color = "#1F77B4", before_color = "GreenYellow",after_color="Orchid", var_boundary_color = 'blue',midline = "red",height = 1500, width=1200, textfont_size = 9):
            assert model_key in self.desc_stats.keys()
            
            match_stat_name = 'match_stats_wts' if weights else 'match_stats'
            combined = self.desc_stats.get(model_key).get('basic_stats').copy()
            title_label = f'Model: {model_key}'
            
            if post_match:
                if custom_match is None:
                    after = self.matched_sets.get(model_key).get(match_key).get(match_stat_name).copy()
                else:
                    after = custom_match
                
                combined = pd.concat([after.assign(time='<b>After Matching</b>'),combined.assign(time="<b>Before Matching</b>")],ignore_index=True)
                title_label += f', Match: {match_key}'
            else:
                combined = combined.assign(time="<b>Before Matching</b>")
                title_label += ', Before Matching'
            
            ##Drop the dist and multivariate one
            combined_reduced = combined.loc[~combined.Covariates.isin(['Multivariate measure','dist'])].reset_index(drop=True).copy()
            num_pattern = '|'.join(self.num_vars+['psm_score','lpsm_score'])
            num_var_bool = combined_reduced.Covariates.str.contains(num_pattern,regex=True)
            vartype_groups = combined_reduced.groupby(num_var_bool)

            ###Plots
            title_label = f'({title_label})' if print_model_info else ''
            
            try:
                cat_df = vartype_groups.get_group(False).reset_index(drop=True)
            except KeyError:
                cat_df=None
                bar_charts = None
            
            if cat_df is not None:
                cat_mean_df = pd.concat([cat_df[['Covariates','N_c','Mean_c','time']].assign(group='<b>Comparison</b>').rename({'Mean_c':'Means','N_c':'N'},axis=1), cat_df[['Covariates','N_tr','Mean_tr','time']].assign(group="<b>Treatment</b>").rename({'Mean_tr':'Means','N_tr':'N'},axis=1)],ignore_index=True)
                
                cat_mean_df['Means'] = cat_mean_df['Means']*100
                
                ###Creating Bar Charts for Categorical Covariates
                bar_charts = px.bar(cat_mean_df,x='time',y='Means', color='group', facet_col='Covariates', facet_col_wrap=4, barmode='group',text='N',labels={'group':"",'time':""},title=f"<b>Balance Breakdown for Categorical Covariates Before and After Matching<br>{title_label} </b>",template='simple_white', category_orders = {'time':['<b>Before Matching</b>','<b>After Matching</b>']},color_discrete_map = {'<b>Comparison</b>':comp_color, '<b>Treatment</b>': treat_color},facet_col_spacing = 0.04,height = height, width=width)
                
                ##Format the text values
                bar_charts.update_traces(texttemplate='<b>%{text}</b>',textfont_size = 9,marker_line_color = 'black',marker_line_width=1,textposition = "outside",selector = dict(type='bar'))
                
                ##Format the subplot titles
                bar_charts.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>',font_size=7))
                
                ##Format the axes
                bar_charts.update_yaxes(title_text="",mirror=True,matches=None,showticklabels=True,tickfont_size=7)
                bar_charts.update_xaxes(title_text="",showticklabels=True,tickfont_size = 7,mirror=True)
                bar_charts.add_annotation(x=-0.05,y=0.5,text="<b>Percent of Total Individuals in Sample Per Group <br>(counts in text)</b>", textangle=-90, xref="paper", yref="paper",font_size=14)
                
                ###Make the y-axis labels dynamically
                
                ##Extract all go.Bar plots and create a key based on the y-axis they associated with
                all_varbar_obj = {graph['yaxis']+'_'+graph['offsetgroup']: graph for graph in bar_charts.data if graph['type']=='bar'}
                
                #Sort them in order so they occur in consecutive pairs, convert to dict and then return list of values without keys
                sorted_varbar_obj = list(dict(sorted(all_varbar_obj.items())).values())
               
                #Loop over the even number positions because the next consecutive one will be the other offsetgroup (time)
                for i in np.arange(0,len(sorted_varbar_obj),2):
                    #Grab both offsetgroups for each axis name
                    tmp_plot_list = sorted_varbar_obj[i:i+2]
                    #Extract the axis number
                    axis_check = re.search('\d+',tmp_plot_list[0]['yaxis'])
                    #Append the number to yaxis so it can be used in the update_layout o identify each axis
                    if axis_check is not None:
                        axis_name = 'yaxis' + axis_check.group()
                    #Case where it is the first axis
                    else:
                        axis_name='yaxis1'
                   
                    #Get array of combined y_vals from each offsetgroup
                    y_vals = np.concatenate([graph['y'] for graph in tmp_plot_list])
                    
                    #Get the max and add 1.5 so labels will show properly outside (necessary for 0) format two decimals places
                    y_max = np.ceil(np.nanmax(y_vals))+10
                    #Force the guide lines to be printed if they fall inside
                    #y_max = 0.5 if y_max<0.25 else y_max
                    
                    ##Create each y-axis based on log_vals but normal labels
                    if y_max<20:
                        y_max = y_max-5
                        y_lab = np.arange(0,y_max,5)
                    elif y_max>100:
                        y_max = y_max+30
                        y_lab = np.arange(0,y_max,25)
                        y_lab = y_lab[y_lab<=100]
                    elif y_max >60:
                        y_max = y_max+20
                        y_lab = np.arange(0,y_max,30)
                    else:
                        y_max=y_max+5
                        y_lab = np.arange(0, y_max,10)
                    
                    bar_charts['layout'][axis_name].update(range=[0, y_max],tickvals = y_lab, ticktext = y_lab)
                
                ##Format the Title and Legend
                bar_charts.update_layout(title_x=0.5,legend_borderwidth=1)
            
            #################Creating Bar Charts for Norm Diff and within diff##############
            mean_cols = ['Covariates','time','Within_match_diff','Norm_diff'] if post_match else ['Covariates','time','Norm_diff']
            mean_df = pd.melt(combined_reduced[mean_cols],id_vars=['Covariates','time']).dropna()
            mean_df.replace({'Within_match_diff':'<b>Within Match</b>','Norm_diff':'<b>Overall Match </b>'},inplace=True)
            mean_bar = px.bar(mean_df,x='variable',y='value',color='time',facet_col = 'Covariates',barmode="group",facet_col_wrap=4,text='value',labels={'time':'','variable':''}, title = f"<b>Absolute Standardized Difference in Means for All Covariates Before and After Matching<br>{title_label}</b>", template='simple_white', category_orders = {'time':['<b>Before Matching</b>','<b>After Matching</b>'],'variable':['<b>Overall Match </b>','<b>Within Match</b>']},color_discrete_map = {'<b>Before Matching</b>':before_color, '<b>After Matching</b>': after_color}, facet_col_spacing = 0.04, height = height, width=width)
            
            ##Add the 0.25 line
            mean_bar.add_trace(go.Scatter(x = ['<b>Within Match</b>','<b>Overall Match </b>'],y=[-1,-1],mode = "lines", line_dash='dot',line_color="red", line_width = 1, name="<b>0.25 theshold</b>",legendgroup="midline", showlegend=True, visible=True,opacity=1),row=1, col=1)
            mean_bar.add_hline(y=0.25, line_dash='dot',line_color="red", line_width = 1, name="0.25 line",row="all", col="all", opacity=1, exclude_empty_subplots=True)
            
            ##Add the 0.1 line
            mean_bar.add_trace(go.Scatter(x = ['<b>Within Match</b>','<b>Overall Match </b>'],y=[-2,-2],mode = "lines", line_dash='dot',line_color="blue", line_width = 1, name="<b>0.10 theshold</b>",legendgroup="midline", showlegend=True, visible=True,opacity=1),row=1, col=1)
            mean_bar.add_hline(y=0.1, line_dash='dot',line_color="green", line_width = 1, opacity=1, name="0.10 line",row="all", col="all", exclude_empty_subplots=True)
            
            ##Round the text values
            mean_bar.update_traces(texttemplate='<b>%{y:.2f}</b>',textfont_size = textfont_size,marker_line_color = 'black',marker_line_width=1,textposition = "outside",selector = dict(type='bar'))
            
            ##Format the subplot titles
            mean_bar.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>',font_size=textfont_size))
            
            ##Format the axes
            mean_bar.update_yaxes(title_text="",mirror=True,tickfont_size=textfont_size,matches=None,showticklabels=True)
            mean_bar.update_xaxes(title_text="",showticklabels=True,tickfont_size = textfont_size,mirror=True,ticklen=0)
            mean_bar.add_annotation(x=-0.05,y=0.5,text='<b>Absolute Standardized Difference in Means</b>', textangle=-90, xref="paper", yref="paper",font_size=14)
            
            ###Make the y-axis labels dynamically
            
            ##Extract all go.Bar plots and create a key based on the y-axis they associated with
            all_varbar_obj = {graph['yaxis']+'_'+graph['offsetgroup']: graph for graph in mean_bar.data if graph['type']=='bar'}
            
            #Sort them in order so they occur in consecutive pairs, convert to dict and then return list of values without keys
            sorted_varbar_obj = list(dict(sorted(all_varbar_obj.items())).values())
           
            #Loop over the even number positions because the next consecutive one will be the other offsetgroup (time)
            for i in np.arange(0,len(sorted_varbar_obj),2):
                #Grab both offsetgroups for each axis name
                tmp_plot_list = sorted_varbar_obj[i:i+2]
                #Extract the axis number
                axis_check = re.search('\d+',tmp_plot_list[0]['yaxis'])
                #Append the number to yaxis so it can be used in the update_layout o identify each axis
                if axis_check is not None:
                    axis_name = 'yaxis' + axis_check.group()
                #Case where it is the first axis
                else:
                    axis_name='yaxis1'
                #Get array of combined y_vals from each offsetgroup
                y_vals = np.concatenate([graph['y'] for graph in tmp_plot_list])
                
                #Get the max and add 1.5 so labels will show properly outside (necessary for 0) format two decimals places
                y_max = np.ceil(np.nanmax(y_vals)*100)/100+0.25
                
                if np.isinf(y_max):
                    #self.logger.info("Here is y_max: ",y_max)
                    self.logger.warning(f'The Bar Graph for {model_key} with match {match_key} may be corrupt due to np.nan')
                    tmp_yvals = y_vals[(y_vals!=np.inf)&(y_vals!=np.NINF)]
                    #self.logger.info("Here are tmp_yvals: ", tmp_yvals)
                    y_max = np.ceil(np.nanmax(tmp_yvals)*100)/100+0.25

                #Force the guide lines to be printed if they fall inside
                y_max = 0.5 if y_max<0.25 else y_max
                
                ##Create each y-axis based on log_vals but normal labels
                
                y_lab = np.arange(0, np.ceil(y_max),0.25)
                y_text = ["<b>" + format(y,".2f") + "</b>" for y in y_lab]
                
                mean_bar['layout'][axis_name].update(range=[0, y_max],tickvals = y_lab, ticktext = y_text)
            
            ##Format the Title and Legend
            mean_bar.update_layout(title_x=0.5,legend_title = '',legend_borderwidth=1
                                   )
            
            ####################Plots for Variance Measures########################!
            var_df = pd.melt(combined_reduced[['Covariates','time','Var_ratio','Resid_ps_var_ratio']],id_vars=['Covariates', 'time'])
            var_df['value_adj'] = np.log2(var_df['value'])
            var_df.replace({'Var_ratio':'<b>Variance Ratio</b>','Resid_ps_var_ratio':'<b>Variance Ratio of<br>Propensity Score<br>Residuals </b>',np.inf:np.nan,-np.inf:np.nan},inplace=True)
            var_df['text_value'] = var_df['value'].map(lambda x: '<b>>32 ({:1.2e})</b>'.format(x) if x>1000 else ('<b>>32 ({:.2f})</b>'.format(x) if x>32 else '<b>{:.2f}</b>'.format(x)))
            
            var_bar = px.bar(var_df,x='variable',y='value_adj',color='time',facet_col = 'Covariates',barmode="group",facet_col_wrap=4,text='text_value',labels={'time':'','variable':''}, title = f"<b>Variance Ratios for All Covariates Before and After Matching<br>{title_label}</b>", template='simple_white', custom_data = ['value'],category_orders = {'time':['<b>Before Matching</b>','<b>After Matching</b>'],'variable':['<b>Variance Ratio</b>','<b>Variance Ratio of<br>Propensity Score<br>Residuals </b>']},color_discrete_map = {'<b>Before Matching</b>':before_color, '<b>After Matching</b>': after_color},facet_col_spacing = 0.04, height = height, width=width)

            ##Add the 1/2 line
            var_bar.add_trace(go.Scatter(x = ['<b>Variance Ratio</b>','<b>Variance Ratio of<br>Propensity Score<br>Residuals </b>'],y=[-100,-100],mode = "lines", line_dash='dot',line_color=var_boundary_color, line_width = 1, name="<b>Variance Ratio Boundaries</b>",legendgroup="midline", showlegend=False, visible=True,opacity=0.8),row=1, col=1)
            var_bar.add_hline(y=-1, line_dash='dot',line_color=var_boundary_color, line_width = 1, name="Ratio = 1/2 line",row="all", col="all", exclude_empty_subplots=True,opacity=0.8)
             
            ##Add the 2 line
            var_bar.add_trace(go.Scatter(x = ['<b>Variance Ratio</b>','<b>Variance Ratio of<br>Propensity Score<br>Residuals </b>'],y=[-300,-300],mode = "lines", line_dash='dot',line_color=var_boundary_color, line_width = 1, name="<b>Variance Ratio Boundaries</b>",legendgroup="midline", showlegend=True, visible=True,opacity=0.8),row=1, col=1)
            var_bar.add_hline(y=1, line_dash='dot',line_color=var_boundary_color, line_width = 1, name="Ratio = 2 line",row="all", col="all", exclude_empty_subplots=True,opacity=0.8)
             
            ##Round the text values
            var_bar.update_traces(textfont_size = textfont_size,marker_line_color = 'black',marker_line_width=1,textposition = "outside",selector = dict(type='bar'))
            
            ##Format the subplot titles
            var_bar.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>',font_size=textfont_size))
            
            ##Format the axes
            var_bar.update_yaxes(title_text="",mirror=True,tickfont_size=textfont_size,matches=None,showticklabels=True)#,type="log",dtick="log_10(2)")
            var_bar.update_xaxes(title_text="",showticklabels=True,tickfont_size = textfont_size,mirror=True,ticklen=0)
            var_bar.add_annotation(x=-0.05,y=0.5,text='<b>Variance Ratio</b>', textangle=-90, xref="paper", yref="paper",font_size=14)
            
            ###Make the y-axis labels dynamically
            
            ##Extract all go.Bar plots and create a key based on the y-axis they associated with
            all_varbar_obj = {graph['yaxis']+'_'+graph['offsetgroup']: graph for graph in var_bar.data if graph['type']=='bar'}
            
            #Sort them in order so they occur in consecutive pairs, convert to dict and then return list of values without keys
            sorted_varbar_obj = list(dict(sorted(all_varbar_obj.items())).values())
           
            #Loop over the even number positions because the next consecutive one will be the other offsetgroup (time)
            for i in np.arange(0,len(sorted_varbar_obj),2):
                #Grab both offsetgroups for each axis name
                tmp_plot_list = sorted_varbar_obj[i:i+2]
                #Extract the axis number
                tmp_axis = tmp_plot_list[0]['yaxis']
                axis_check = re.search('\d+',tmp_axis)
                #Append the number to yaxis so it can be used in the update_layout o identify each axis
                if axis_check is not None:
                    axis_name = 'yaxis' + axis_check.group()
                #Case where it is the first axis
                else:
                    axis_name='yaxis1'
                
                #Get array of combined y_vals from each offsetgroup
                y_vals1 = tmp_plot_list[0]['y']
                y_vals2 = tmp_plot_list[1]['y']
                
                y_vals = np.concatenate([y_vals1,y_vals2])
                
                if all(np.isnan(y_vals)):
                    
                    var_bar['layout'][axis_name].update(range = [-1.75,1.75],tickvals=[-1,0,1],ticktext= ["<b>" + format(y,".2f") + "</b>" for y in [0.5,1,2]])
                    continue
                #Get the min and subtract 1.5 so labels will show properly outside (necessary for 0) format two decimal places
                y_min = np.floor(np.nanmin(y_vals)*100)/100-1.5
                #Force the guide lines to be printed if they fall inside
                y_min =  -1.75 if y_min>-1 else y_min
                
                #Get the max and add 1.5 so labels will show properly outside (necessary for 0) format two decimals places
                y_max_orig = np.nanmax(y_vals)
                y_max = np.ceil(y_max_orig*100)/100+1.5
                #Force the guide lines to be printed if they fall inside
                y_max = 1.75 if y_max<1 else y_max
                    
                ##Create each y-axis based on log_vals but normal labels
                y_log_lab = np.arange(np.floor(y_min),np.ceil(y_max))
                y_lab = np.round(np.exp2(y_log_lab),2)
                y_text = ["<b>" + format(y,".2f") + "</b>" for y in y_lab]
                
                ##Create logic for large values above 32 (2^5)
                if y_max_orig>5:
                    large_yvals, = np.where(y_vals>5)
                    if any(large_yvals<2):
                       
                        var_bar.update_traces(textposition='auto', insidetextanchor = "start", outsidetextfont_size= textfont_size, selector=dict(type='bar', offsetgroup=tmp_plot_list[0]['offsetgroup'], yaxis = tmp_axis))
                    if any(large_yvals>1):

                        var_bar.update_traces(textposition='auto',insidetextanchor = "start", outsidetextfont_size= textfont_size, selector=dict(type='bar', offsetgroup=tmp_plot_list[1]['offsetgroup'], yaxis = tmp_axis))
                        
                    y_log_lab = np.arange(np.floor(y_min),6)
                    y_lab = np.round(np.exp2(y_log_lab),2)
                    y_max = 5
                    y_text = ["<b>" + format(y,".2f") + "</b>" for y in y_lab]
                
                var_bar['layout'][axis_name].update(range=[y_min, y_max],tickvals = y_log_lab, ticktext = y_text)
               
            ##Format the Title and Legend
            var_bar.update_layout(title_x=0.5,legend_title = '',legend_borderwidth=1)
            
            return bar_charts, mean_bar, var_bar"""

        st.code(plotly_covar_balance,language="python")
        
        
if selected=='Phase 4':
    st.title("Phase 4: Performing Outcome Analysis on Matched Set")
    
    st.markdown('#### After the best matched set was chosen, which in our case was exact matching, we performed a weighted GLM regression. AD status was regressed on the HSV-1 indicator using the weights described in phase 3</sup><sup class="sup">12,</sup></sup><sup class="sup">16,</sup></sup><sup class="sup">17</sup>. Because exact matching was used, we did not have to control for matching covariates in our primary model.',unsafe_allow_html=True)
    
    st.markdown('#### Code examples are shown below for different ways of performing the final analysis: ')
    
    st.code('''#Weighted GLM of AD on HSV-1 status with no covariates needed due to exact matching
my_psm.perform_outcome_analysis("psmodel1","nearest3",test="glm",weighted=True,covar_selection="treat")
#Weighted GLM using all covariates that were unbalanced based on phase 3 criteria with AD as the outcome and HSV-1 status as the treatment
my_psm.perform_outcome_analysis("psmodel1","nearest4",test="glm",weighted=True,covar_selection="auto")
''')
    st.markdown('#### Expand below if interested in the main outcome function and its helper functions :')
    
    with st.expander('##### Main Function: perform_outcome_analysis'):
        st.markdown('''*This function currently performs McNemars test for binary outcomes without covariates and glm with both weighted and unweighted depending on whether full matching, variable matching or constant matching*
        
        1. test can be mcnemar, glm
        
    2. covar selection can be auto, all, manual
    
    3. covar_threshold is float based on max balance distance required to add covar to glm
    
    4. covars is for manually specifiying the covariates and overrides covar selection and covar threshold, extra vars can be used if variable is under threshold but you may want to include in regression
    
    5. Adding t-test and linear regression (lm) for continous outcomes''')
        
        perform_outcome_analysis = """def perform_outcome_analysis(self,model_key, match_key, test = "mcnemar", covar_selection="auto", covar_threshold = 0.05, covar_stat_type = 'overall',match_covars=None, other_covars = None, weighted = False,display=True):
    
    assert covar_stat_type in ['overall','within']
    if not self.outcome_analysis_valid:
        self.logger.exception('You must specify a target variable to use this method!')
        raise NotImplementedError
    if other_covars is not None:
        assert isinstance(other_covars,list)
        assert all([c in self.masterDf.columns for c in other_covars])
        if match_covars is not None:
            other_covars_clean = list(np.setdiff1d(other_covars, match_covars))
        else:
            other_covars_clean = other_covars
        
    df = self.select_matched_dataset(model_key, match_key,extra_vars = other_covars)
    if df is None:
        self.logger.warning(f'Could not perform outcome analysis because there are no matches for {match_key}_{model_key}')
        return None
    if test.lower().startswith("mcn") or test.lower().startswith("ttest") or test.lower().startswith("wilcox"):
        mc_df = df[['group_id',self.target_var, self.treat_var]]
        #Average if one to many matching
        mc_df = mc_df.groupby(['group_id',self.treat_var]).mean().reset_index()
        mc_df = mc_df.pivot(index="group_id",columns=self.treat_var)
        
        if test.lower().startswith("mcn"):
            mc_df = mc_df.apply(lambda x: np.where(np.round(x)==0,"No","Yes"),axis=0)
            mc_df.columns = ['comparison','treatment']
            mcn_exact = False
            if len(mc_df)<20:
                mcn_exact = True
            contingency_table = pd.crosstab(mc_df['comparison'],mc_df['treatment'])
            mcnemar_test = mcnemar(contingency_table,exact=mcn_exact)
            if display:
                self.logger.info(mcnemar_test)
            return contingency_table, mcnemar_test
        elif test.lower().startswith("ttest"):
            mc_df.columns = ['comparison','treatment']
            t_test = stats.ttest_rel(mc_df['comparison'], mc_df['treatment'])
            if display:
                self.logger.info(t_test)
            return t_test
        elif test.lower().startswith("wilcox"):
            mc_df.columns = ['comparison','treatment']
            wilcox = stats.wilcoxon(mc_df['comparison'], mc_df['treatment'])
            if display:
                self.logger.info(wilcox)
            return wilcox
    elif test.lower()=='glm' or test.lower()=='lm':
        #Check covars first because if so then will override all else
        
        if match_covars is None:
            match_stat_name = 'match_stats_wts' if weighted else 'match_stats'
            stat_col = 'Norm_diff' if covar_stat_type=='overall' else 'Within_match_diff'
            balance_stats = self.matched_sets[model_key][match_key].get(match_stat_name)
            if balance_stats is None:
                print(f'You need to run the assess matches method with the parameter, "weights={weighted}" for this mmatched set first')
                return None
            if covar_selection =='auto':
                match_covars = balance_stats.loc[(balance_stats[stat_col]>covar_threshold) & ~(balance_stats['Covariates'].isin(['lpsm_score','psm_score','dist','Multivariate measure'])),'Covariates'].to_numpy().tolist()
                if len(match_covars)==0:
                    if other_covars is None:
                        self.logger.warning(f"All covariates are balanced based on the specified threshold of {covar_threshold} for {model_key}_{match_key}. Either set the argument of covar_selection to 'all' or 'treat', specify the names of match_covars manually, specify extra vars, or use the McNemar's Test instead!")
                        return None
                covars = match_covars
            
            elif covar_selection=="all":
                covars = balance_stats.Covariates[(balance_stats['Norm_diff'].notnull()) & ~(balance_stats.Covariates.isin(['lpsm_score','psm_score','dist',"Multivariate measure"]))].to_numpy().tolist()   
               
            elif covar_selection=="treat":
                covars = []
            
            else:
                self.logger.exception('Argument for covar_selection method can only be "auto","treat", or "all"!')
                raise ValueError
        else:
            covars = match_covars
        if other_covars is not None:
            covars = covars + other_covars_clean
            
            covars = self._check_user_category_vars(covars)
        
        X1 = sm.add_constant(df[[self.treat_var] + covars])
       
        ###Check for binary outcome or continuous
        glm_family_type = sm.families.Gaussian() if self.target_var_type=='numeric' else sm.families.Binomial()
        
        ##Check for exact_only
        if self.matched_sets[model_key][match_key]['exact_only']:
            if weighted:
                model_results = sm.GLM(df[self.target_var], X1, family = glm_family_type, freq_weights = df['ATT_wts']).fit(disp=display)
                #self.logger.info("This is working!!!!")
            else:
                model_results = sm.GEE(df[self.target_var], X1, family = glm_family_type, groups=df["group_id"]).fit()
            if display:
                self.logger.info(model_results.summary())
            return model_results
        
        ###Check for w/ replacement
        match_method = self.matched_sets[model_key][match_key]['match_method']
        
        if match_method=='nearest' and self.matched_sets[model_key][match_key]['replace']:
            #if self.matched_sets[model_key][match_key]['replace']:
            model_results = sm.GEE(df[self.target_var], X1, family = glm_family_type, groups=df["w_replace_id"]).fit()
            if display:
                print(model_results.summary())
            return model_results
        else:
        
            ###Check for weighted or unweighted
            if weighted:
                model_results = sm.GLM(df[self.target_var], X1, family = glm_family_type, freq_weights = df['ATT_wts']).fit(disp=display)
            else:
                model_results = sm.GLM(df[self.target_var], X1, family = glm_family_type).fit(disp=display)
        if display:
            self.logger.info(model_results.summary())
        return model_results """
        
        st.code(perform_outcome_analysis,language="python")
        
    with st.expander('##### Helper 1: select_matched_dataset'):
        select_matched_dataset = """def select_matched_dataset(self, model_key, match_key, extra_vars = None, all_indicators=True):
            
    comb_df = self._combine_matches_with_data(model_key, match_key,return_df=True,extra_vars = extra_vars, weights=True, all_indicators = all_indicators)
    if comb_df.shape[0]==0:
        self.logger.warning(f'No matches found for {match_key} under {model_key}!')
        return None
    
    if self.matched_sets[model_key][match_key]['exact_only']:
        return comb_df
    
    if self.matched_sets[model_key][match_key]['match_method']=='optimal':
        return comb_df.drop_duplicates('match_id').reset_index(drop=True)

    if self.matched_sets[model_key][match_key]['replace']:
        groups = comb_df.groupby(self.treat_var)
        comps = groups.get_group(0)
        treats = groups.get_group(1)
        ###This first selects all the groups_ids for each unique control. This array of group ids is then passed into the treatment to match the group id with the treatment id. The result is a dictionary with each key a unique comp id and the value an array of match_ids including the comp_id
        comp_gid_dict =dict(map(lambda x: (x,np.append(treats.loc[treats.group_id.isin(comps.loc[comps['match_id']==x,'group_id'].to_numpy()),'match_id'].to_numpy(),x)),comps.match_id.unique()))
        
        ##This creates a list of arrays with col1 each unique match_id and col2 a replace_id that is the same as the unique comp key for error checking
        comp_gid = pd.DataFrame(np.vstack([np.c_[v,np.repeat(k,len(v))] for k,v in comp_gid_dict.items()]),columns=['match_id','w_replace_id'])
        return pd.merge(comp_gid,comb_df.drop_duplicates(subset=['match_id']), on='match_id',how="left")

    return comb_df
 """
        
        st.code(select_matched_dataset,language="python")
    
    with st.expander('##### Helper 2: get_psm_df'):
        
        st.markdown( '''*This function retrieves the psm df by threshold and then order in list which corresponds to string of variables. This list of variables string corresponds to the key in ps_estimators in order to access the ps_scores. These can then be converted into linearized scores. The df is then assembled by making the first three columns the match_id, target_var, and treat_var respectively. These are followed by the covariates contained in the string key and then lastly the ps_score and lpsm_Score. Any extra vars to include for later analysis but not for matching will be added after the first three but before the covariates so will just need to start at higher index in matching functions. This function can be called with pval_key directly with manual_pval_key or looked up using the first and second order threshold if made using ML approach*''')
        
        get_psm_df = """ def get_psm_df(self,index, custom = False, first_order_pval = stats.norm.sf(1), second_order_pval = 0.05,manual_pval_key = None, extra_vars = None, allow_refs = True, all_indicators = False, stats=False):
    #Get all model keys in order of best LR per p-value threshold
    all_model_keys = self.get_all_best_models(print_results=False)
    if custom:
        if "custom" not in all_model_keys:
            self.logger.exception('You have not entered any custom models yet!')
            raise KeyError
        pval_key = "custom"
    #Case if df was made using automatic selection with thresholds
    else:
        #Using the threshold look up to get pval_key
        if manual_pval_key is None:
            pval_key = self._convert_pval_key(first_order_pval,second_order_pval)
        #If pval_key is known
        else:
            pval_key = manual_pval_key
    
    if pval_key not in all_model_keys:
        self.logger.exception(f'You have not run any models under the following threshold key: {pval_key}')
        raise KeyError
    
    model_key = self.best_model_meta[pval_key][index][0]
   
    if extra_vars is not None:
        if isinstance(extra_vars, str):
            extra_vars = [extra_vars]
        else:
            assert isinstance(extra_vars, (list, tuple, np.ndarray))
        ##Need to combine extra_vars to model vars to prevent duplication
        all_vars_comb = extra_vars +  literal_eval(model_key)
       
        clean_vars = self._check_user_category_vars(all_vars_comb, allow_refs)
        extra_vars = list(np.setdiff1d(clean_vars, literal_eval(model_key)))
        
    else:
        extra_vars = []
        
    if all_indicators:
        is_binary = self.best_model_meta[pval_key][index][1]
        ##If at least one cat var need to check indicator vars
        if len(self.cat_vars)>0:
            all_ind_dict = self._get_all_indicators(literal_eval(model_key),is_binary=is_binary)
            new_clean_vars = list(all_ind_dict.keys())
            sorted_col_ind = np.argsort(new_clean_vars)
            new_clean_vars = list(np.take(np.array(new_clean_vars),sorted_col_ind))
            is_binary = list(np.take(np.array(list(all_ind_dict.values())),sorted_col_ind))
            ##Handle any extra vars. Could include extra vars that are included in all indicator vars
            extra_vars = list(np.setdiff1d(extra_vars, new_clean_vars))
            if len(extra_vars)>0:
                #If no numeric than all extra vars must be binary
                if len(self.num_vars)==0:
                    extra_var_bin = [True]*len(extra_vars)
                else:
                    num_pat = "|".join(self.num_vars)
                    #Split on interactions and explode
                    extra_vars_long = pd.Series(extra_vars).str.split(".").explode()
                    num_extra_vars = extra_vars_long[extra_vars_long.str.contains(num_pat, regex=True)].index
                    extra_var_bin = [False if x in num_extra_vars else True for x in extra_vars]
            #No extra vars - need to make a extra bin placeholder    
            else:
                extra_var_bin = []
            
            master_ps_cols = [self.match_id, self.target_var, self.treat_var] + extra_vars + new_clean_vars
            #Need to maintain order of is-Binary to be compatible later
            is_binary = [True]*3 + extra_var_bin + is_binary + [False]*2
        
        #This means no binary variables exist in df
        else:
            master_ps_cols = [self.match_id, self.target_var, self.treat_var] + extra_vars +  literal_eval(model_key)
            is_binary = [True]*3 +[False]*(len(master_ps_cols)-3)
    else:
        master_ps_cols = [self.match_id, self.target_var, self.treat_var] + extra_vars +  literal_eval(model_key)
        is_binary = self.best_model_meta[pval_key][index][1]
    
    psm_df = self.masterDf[master_ps_cols].copy()
    
    psm_df['psm_score'] = self.ps_estimators[pval_key][model_key][1]
    
    ####Check for inf values which will occur if psm scores close to 1! A psm_score of 0.9999999999999998 will correspind to a lpsm of 36 for 16 digit precision.
    model_info = 'during creation of the linearized propensity score for assembling the dataframe'
    psm_df['lpsm_score'] = self._calc_linear_psm(model_key, model_info,self.ps_estimators[pval_key][model_key][1])
    pos_inf_vals, = np.where(np.isclose(psm_df['psm_score'],1,atol=1e-16,rtol=1e-16))
    if len(pos_inf_vals)>0:
        self.logger.warning(f'Propensity score of 1 detected in {model_key}, which will result in an infinite linear propensity score! Automatically assigning 40 to such values but best to match on actual propensity score or use a better logistic regression model!')
        psm_df.iloc[pos_inf_vals,-1] = 40
    neg_inf_vals, = np.where(np.isclose(psm_df['psm_score'],0,atol=1e-323,rtol=1e-323))
    if len(neg_inf_vals)>0:
        self.logger.warning(f'Propensity score of 0 detected in {model_key}, which will result in a negative infinite linear propensity score! Automatically assigning -745 to such values but best to match on actual propensity score or use a better logistic regression model!')
        psm_df.iloc[neg_inf_vals,-1] = -745
        
    if stats:
        return psm_df,is_binary
    return psm_df"""
        
        st.code(get_psm_df,language="python")
    
    
if selected=='References':
    st.title('References')
    
    st.markdown("[1] Austin PC. Comparing paired vs non-paired statistical methods of analyses when making inferences about absolute risk reductions in propensity-score matched samples. Stat Med. 2011;30(11):1292-1301.")
    
    st.markdown("[2] Rubin DB. Estimating causal effects of treatments in randomized and nonrandomized studies. J Educ Psychol. 1974;66(5):688-701.")
    
    st.markdown("[3] Stuart, E. A. 2010. Matching Methods for Causal Inference: A Review and a Look Forward. Statistical Science 25(1), 1–21.")
    
    st.markdown("[4] Austin PC. Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. Stat Med. 2009;28(25):3083-3107.")

    st.markdown('[5] Imbens GW, Rubin DB. Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction. Cambridge University Press; 2015.')
    
    st.markdown('[6] Rosenbaum PR. Design of Observational Studies. Springer New York.')
    
    st.markdown('[7] Austin PC. Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. Stat Med. 2009;28(25):3083-3107.')
    
    st.markdown('[8] Austin PC. The performance of different propensity score methods for estimating marginal odds ratios. Stat Med. 2007;26(16):3078-3094.')
    
    st.markdown('[9] Imai K, King G, Stuart EA. Misunderstandings among Experimentalists and Observationalists about Causal Inference. https://papers.ssrn.com › sol3 › papershttps://papers.ssrn.com › sol3 › papers. Published online December 20, 2007. Accessed December 30, 2022. https://papers.ssrn.com/abstract=1013351')
    
    st.markdown('[10] Rubin DB. Using Propensity Scores to Help Design Observational Studies: Application to the Tobacco Litigation. Health Serv Outcomes Res Methodol. 2001;2(3):169-188.')
    
    st.markdown('[11] Rosenbaum PR, Rubin DB. Constructing a Control Group Using Multivariate Matched Sampling Methods That Incorporate the Propensity Score. Am Stat. 1985;39(1):33-38.')
    
    st.markdown('[12] Green KM, Stuart EA. Examining moderation analyses in propensity score methods: application to depression and substance use. J Consult Clin Psychol. 2014;82(5):773-783.')
    
    st.markdown('[13] King G, Nielsen R. Why Propensity Scores Should Not Be Used for Matching. Polit Anal. 2019;27(4):435-454.')
    
    st.markdown('[14] Harder VS, Stuart EA, Anthony JC. Propensity score techniques and the assessment of measured covariate balance to test causal associations in psychological research. Psychol Methods. 2010;15(3):234-249.')
    
    st.markdown('[15] Guo S, Fraser M, Chen Q. Propensity Score Analysis: Recent Debate and Discussion. J Soc Social Work Res. 2020;11(3):463-482.')
    
    st.markdown('[16] Stuart EA, Green KM. Using full matching to estimate causal effects in nonexperimental studies: examining the relationship between adolescent marijuana use and adult outcomes. Dev Psychol. 2008;44(2):395-406.')
    
    st.markdown('[17] Hansen BB. Full Matching in an Observational Study of Coaching for the SAT. J Am Stat Assoc. 2004;99(467):609-618.')

    












