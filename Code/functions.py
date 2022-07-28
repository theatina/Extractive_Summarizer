import os
import json
import pandas as pd
import re


def json_to_df(json_path,type):
  with open(json_path, "r", encoding="utf-8") as f: 
    lines = [eval(l) for l in f.readlines()]

  # exclude lines with surrogates in their text/summary
  surr = [ i for i,l in enumerate(lines) for k in l.keys() if k in ["text","summary"] and re.search(r'[\uD800-\uDFFF]', l[k])!=None ]
          
  lines = [ l for i,l in zip( range(len(lines)),lines ) if i not in surr ]

  cols=[ "title",	"date",	"text",	"summary", "compression", "coverage", "density", "compression_bin", "coverage_bin"]

  # we need only the extractive summaries as we are building an extractive summarizer
  data=[ [ l[k] for k in l.keys() if k in cols ] for l in lines if l["density_bin"]=="extractive" ]
  df = pd.DataFrame(data,columns=cols)

  df.to_csv(f"..{os.sep}Data{os.sep}DataFrames{os.sep}{type}_set.csv", header=True, index=False )

  return df