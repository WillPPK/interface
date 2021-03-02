import sys
import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import json
from scipy.spatial import cKDTree
import time
import sys
import pylab
import seaborn as sb
import matplotlib.axis as axe
from matplotlib import rc, rcParams
from matplotlib.pyplot import *

def max_concurrent(array, val=1, window=2):
    count = 0
    max_count = 0
    gap = 0
    for t in array + [None]*(window+1):
        if t == val:
            count += 1
            gap = 0
        else:
            gap += 1
        if gap > window:
            if count > max_count:
                max_count = count
            count = 0
    return max_count

class ContactGenerator(object):
    def __init__(self, universe):
        self.universe = universe
    def build_contacts(
            self,
            ligand_selection,
            protein_selection,
            cutoff,
            frameskip,
            KDTree=True):
        protein = self.universe.select_atoms(protein_selection)
        ligand = self.universe.select_atoms(ligand_selection)
        protein_details = []
        ligand_details = []
        for atom in protein:
            protein_details.append(atom.index)
        for atom in ligand:
            ligand_details.append(atom.index)
        frames = {}
        done = 0
        todo = len(self.universe.trajectory) / frameskip
        for ts in self.universe.trajectory[::frameskip]:
            if KDTree:
                l_kdtree = cKDTree(ligand.positions)
                p_kdtree = cKDTree(protein.positions)
                kq = l_kdtree.query_ball_tree(p_kdtree, r=cutoff)
                contacts = []
                for idx, vs in enumerate(kq):
                    contacts += [(idx, v) for v in vs]
                frames[ts.time] = Counter(contacts)
            else:
                frames[ts.time] = Counter([
                    tuple(x)
                    for x in np.argwhere(
                        cdist(ligand.positions, protein.positions) < cutoff
                        )])
            done += 1
            print(done,"    /   ", todo)
        return Contacts(self, ligand_details, protein_details, frames)

class Contacts(object):
    def __init__(self, generator, ligand, protein, frames):
        self.generator = generator
        self.protein = protein
        self.ligand = ligand
        self.frames = frames
        self.unaggregate()
        self.n_frames = len(self.frames)
    def unaggregate(self):
        groupable      = ["resid", "resname", "index", "name", "segid"]
        self.group_protein_by = "index"
        self.group_ligand_by  = "index"
        self.protein_groups = {
            idx : {
                k : getattr(self.generator.universe.atoms[atom_id], k) for k in groupable
            } for idx, atom_id in enumerate(self.protein)}
        self.ligand_groups = {
            idx : {
                k : getattr(self.generator.universe.atoms[atom_id], k) for k in groupable
            } for idx, atom_id in enumerate(self.ligand)}
        self.grouped_frames = self.frames
    def set_group_attr(self, attrname, function, protein=False, ligand=False):
        if protein:
            for group in self.protein_groups:
                self.protein_groups[group][attrname] = function(self.protein_groups[group])
        if ligand:
            for group in self.ligand_groups:
                self.ligand_groups[group][attrname] = function(self.ligand_groups[group])
    def aggregate(
            self,
           group_protein_by=None,
            group_ligand_by=None,
            aggregate_function=lambda x : x.sum() ):
        if group_protein_by is not None:
            self.group_protein_by = group_protein_by
        if group_ligand_by is not None:
            self.group_ligand_by  = group_ligand_by
        # Value : GroupID
        new_protein_group_values = {}

        # Old GroupID : New GroupID
        new_protein_group = {}

        # GroupID : Attributes
        new_protein_group_attributes = {}

        for k, v in self.protein_groups.items():
            if group_protein_by is not None:
                new_protein_group_value = v[group_protein_by]
            else:
                new_protein_group_value = k
            if new_protein_group_value not in new_protein_group_values:
                new_group_id = len(new_protein_group_values)
                new_protein_group_values[new_protein_group_value] = new_group_id
                new_protein_group_attributes[new_group_id] = v
            else:
                new_group_id = new_protein_group_values[new_protein_group_value]
            new_protein_group[k] = new_protein_group_values[new_protein_group_value]
            delkeys = set()
            for k in new_protein_group_attributes[new_group_id]:
                if new_protein_group_attributes[new_group_id][k] != v[k]:
                    delkeys.add(k)
            for k in delkeys:
                del new_protein_group_attributes[new_group_id][k]

        # Value : GroupID
        new_ligand_group_values = {}

        # Old GroupID : New GroupID
        new_ligand_group = {}

        # GroupID : Attributes
        new_ligand_group_attributes = {}

        for k, v in self.ligand_groups.items():
            if group_ligand_by is not None:
                new_ligand_group_value = v[group_ligand_by]
            else:
                new_ligand_group_value = k
            if new_ligand_group_value not in new_ligand_group_values:
                new_group_id = len(new_ligand_group_values)
                new_ligand_group_values[new_ligand_group_value] = new_group_id
                new_ligand_group_attributes[new_group_id] = v
            else:
                new_group_id = new_ligand_group_values[new_ligand_group_value]
            new_ligand_group[k] = new_ligand_group_values[new_ligand_group_value]
            delkeys = set()
            for k in new_ligand_group_attributes[new_group_id]:
                if new_ligand_group_attributes[new_group_id][k] != v[k]:
                    delkeys.add(k)
            for k in delkeys:
                del new_ligand_group_attributes[new_group_id][k]

        new_grouped_frames = {}

        for time, frame in self.grouped_frames.items():
            new_grouped_frame = {}
            for (ligand_group, protein_group), value in frame.items():
                value = 1. * value
                l_group = new_ligand_group[ligand_group]
                p_group = new_protein_group[protein_group]
                if (l_group, p_group) not in new_grouped_frame:
                    new_grouped_frame[(l_group, p_group)] = [value]
                else:
                    new_grouped_frame[(l_group, p_group)].append(value)
            new_grouped_frames[time] = new_grouped_frame
        for time, frame in new_grouped_frames.items():
            for (ligand_group, protein_group), values in frame.items():
                v = np.array(values)
                new_grouped_frames[time][(ligand_group, protein_group)] = aggregate_function(v)

        self.grouped_frames = new_grouped_frames
        self.protein_groups = new_protein_group_attributes
        self.ligand_groups = new_ligand_group_attributes
        return self
    def to_dict(self, key_order=["time", "ligand", "protein"], lookup=True):
        expanded = {}
        for time, frame in self.grouped_frames.items():
            current = {"time" : time}
            for (ligand_group, protein_group), value in frame.items():
                if lookup:
                    lig = self.ligand_groups[ligand_group][self.group_ligand_by]
                    pro = self.protein_groups[protein_group][self.group_protein_by]
                else:
                    lig = ligand_group
                    pro = protein_group
                current["ligand"] = lig
                current["protein"] = pro
                k0 = current[key_order[0]]
                k1 = current[key_order[1]]
                k2 = current[key_order[2]]
                if k0 not in expanded:
                    expanded[k0] = {}
                if k1 not in expanded[k0]:
                    expanded[k0][k1] = {}
                expanded[k0][k1][k2] = value
        return expanded
    def time_aggregate(self, aggregate_function=lambda x : np.sum(x.values())):
        d = self.to_dict(key_order=["ligand", "protein","time"], lookup=False)
        for l in d:
            for p in d[l]:
                d[l][p] = aggregate_function(d[l][p])
        return ContactsTimeGrouped(
            self.ligand_groups,
            self.protein_groups,
            self.group_ligand_by,
            self.group_protein_by,
            d)

class ContactsTimeGrouped(object):
    def __init__(self, ligand_groups, protein_groups, group_ligand_by, group_protein_by, data):
        self.ligand_groups    = ligand_groups
        self.protein_groups   = protein_groups
        self.group_protein_by = group_protein_by
        self.group_ligand_by  = group_ligand_by
        self.data = data
    def to_dict(self, key_order=["ligand", "protein"], lookup=True):
        expanded = {}
        for ligand_group in self.data:
            for protein_group, value in self.data[ligand_group].items():
                current = {}
                if lookup:
                    lig = self.ligand_groups[ligand_group][self.group_ligand_by]
                    pro = self.protein_groups[protein_group][self.group_protein_by]
                else:
                    lig = ligand_group
                    pro = protein_group
                current["ligand"] = lig
                current["protein"] = pro
                k0 = current[key_order[0]]
                k1 = current[key_order[1]]
                if k0 not in expanded:
                    expanded[k0] = {}
                if k1 not in expanded[k0]:
                    expanded[k0][k1] = value
        return expanded
    def to_dataframe(self, key_order=["ligand", "protein"], lookup=True):
        return pd.DataFrame(self.to_dict(key_order=key_order, lookup=lookup)).fillna(0)

U = mda.Universe(sys.argv[1],sys.argv[2])

generate = ContactGenerator(U)

datas_seg = []

for sel_segid in ['A'] : #change to your chain here
	datas = np.zeros([400,1600]) #put the TOTAL number of residue from chain A and a total number of residue of chain E. This can be over - is just making an array
	for sel_resid in np.arange(1,32) : #I make my life easier as I only care about residues 1 to 32 on segid A
			contacts = generate.build_contacts(ligand_selection='segid E', protein_selection='segid ' + str(sel_segid), frameskip=1,cutoff=4,KDTree=True) #this will be other chain - I put E here as my protein is a tetramer. But change to whatever. Cutoff = 4. Change frameskip if you want. I recommend you read less than 1000 frames.
			contacts.aggregate(group_protein_by="resid",group_ligand_by="resname",aggregate_function=lambda x:x.max())
			data = contacts.time_aggregate(aggregate_function=lambda x:sum(x.values())/contacts.n_frames)
			for k, j in data.to_dataframe().iterrows() :
				datas[sel_resid][k] =  j[0]		
datas_seg.append(datas)
		
#Save data as matrix
datas_seg = np.array(datas_seg)
datas_av = np.average(datas_seg, axis = 0)
np.savetxt(sys.argv[3] + ".dat",datas_av) 

#Show as list
df = pd.DataFrame(datas_av).unstack()
df.index.names = ['ref', 'resid']
df = df.reset_index()
df.to_csv(sys.argv[3]+".csv")

#make the graph

data = np.loadtxt(sys.argv[3]+".dat").reshape([400,1600]) #put the TOTAL number of residue from chain A and a total number of residue of chain E. This will help you to shape your map.

data_s = data[1:381,1:1600] #Specifying info to feed to the axis here

v = [1, 381, 1600, 1] #Specifying axis here it reads xmin, xmax, ymax, ymin 

plt.imshow(data_s, cmap=plt.cm.get_cmap('RdBu',8), aspect = 'equal', vmin = -1, vmax = 1, extent=v)

plt.colorbar()
plt.xlabel('Residues')
plt.ylabel('Residues')

plt.show()



