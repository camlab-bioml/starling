#samples = expand("Rscript DAMM/new/code/pro/metrics.R --cohort {cohort}", cohort = COHORTS)
cohorts = ['eddy', 'meta', 'basel']

rule all:
    input:
        expand("{cohort}-metrics.txt", cohort = cohorts)
rule experiment:
    params:
        cohort = '{cohort}'
    output:
        fn = "{cohort}-metrics.txt"
    shell:
        "Rscript DAMM/new/code/pro/metrics.R --cohort {params.cohort} > {output.fn}"