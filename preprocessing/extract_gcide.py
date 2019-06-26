from __future__ import print_function
import sys
import re
import zipfile
import string
import os.path
from collections import defaultdict

### Download gcide_xml-0.51.zip from web
if not os.path.isfile('gcide_xml-0.51.zip'):
  if sys.version_info[0] == 2:
    from urllib import urlretrieve
  else:
    from urllib.request import urlretrieve
  urlretrieve('http://www.ibiblio.org/webster/gcide_xml-0.51.zip', 'gcide_xml-0.51.zip')

### Parse the data
ent_re = re.compile(r'<ent>(.*?)</ent>')
hw_re = re.compile(r'<hw>(.*?)</hw>')
sn_re = re.compile(r'<sn>([0-9]+).</sn>')
def_re = re.compile(r'<def>(.*?)</def>')
syn_re = re.compile(r'<syn><b>Syn. --</b> (.*?)</syn>')
my_regs = [ent_re, hw_re, sn_re, def_re, syn_re]

def search_and_ret(reg):
  m = reg.search(line)
  return m.group(1) if m else None
  
synset_map = {}
def_cnt = defaultdict(lambda: 0)

with zipfile.ZipFile('gcide_xml-0.51.zip', 'r') as zip_file:

  with open('gcide_def.txt', 'w') as def_file:

    for file_name in ['gcide_xml-0.51/xml_files/gcide_{}.xml'.format(x) for x in string.ascii_lowercase]:
    # for file_name in ['gcide_xml-0.51/xml_files/gcide_{}.xml'.format(x) for x in 'a']:
    
      with zip_file.open(file_name, 'r') as my_xml:
    
        last_ent = None
        lines = []
        for line in my_xml:
          line = line.decode('utf-8').strip()
          if line != '':
            lines.append(line)
          else:
            # Find all the values
            line = ''.join(lines)
            lines = []
            my_ent, my_hw, my_sn, my_def, my_syn = [search_and_ret(x) for x in my_regs] 
            # Carry over the last entity if necessary
            if my_ent == None:
              my_ent = last_ent
            else:
              last_ent = my_ent
            # If an entity is found, print it
            if my_ent and my_def and not (' ' in my_ent):
              def_cnt[my_ent] += 1
              my_sn = def_cnt[my_ent]
              print("{}%{} ||| {}".format(my_ent, my_sn, my_def), file=def_file)
              found = False
              if my_syn != None:
                for syn in my_syn.split(', '):
                  my_id = '{} ||| {}'.format(my_syn, my_def)
                  if my_id in synset_map:
                    synset_map[my_id].append('{}%{}'.format(my_ent,my_sn))
                    print('found {}%{} in synset {}'.format(my_ent,my_sn,my_id))
                    found = True
                    break
              if not found:
                my_id = '{} ||| {}'.format(my_ent, my_def)
                synset_map[my_id] = ['{}%{}'.format(my_ent,my_sn)]
                print('adding {}%{} as synset {}'.format(my_ent,my_sn,my_id))
            else:
              print('nothing found in {}'.format(line))
 
with open('gcide_ont.txt', 'w') as ont_file:
  for k, v in synset_map.items():
    if len(v) > 1:
      for x in v:
        vals = ['{}%1.0'.format(x)]
        for y in v:
          if x != y:
            vals.append(['{}%1.0'.format(y)])
        print(vals, file=ont_file)
