#!/usr/bin/python3

import h5py
import numpy as np
import sys


try:
    from lxml import etree
    print("running with lxml.etree")
except ImportError:
    try:
        # Python 2.5
        import xml.etree.cElementTree as etree
        print("running with cElementTree on Python 2.5+")
    except ImportError:
        try:
            # Python 2.5
            import xml.etree.ElementTree as etree
            print("running with ElementTree on Python 2.5+")
        except ImportError:
            try:
                # normal cElementTree install
                import cElementTree as etree
                print("running with cElementTree")
            except ImportError:
                try:
                    # normal ElementTree install
                    import elementtree.ElementTree as etree
                    print("running with ElementTree")
                except ImportError:
                    print("Failed to import ElementTree from any known place")

if len(sys.argv) != 3:
    print("Usage: " + sys.argv[0] + " infile(.hdf5) outfile(.xdmf)")
    sys.exit()

h5filename = sys.argv[1] #"data.h5"
xdmffilename = sys.argv[2] #"data.xdmf"

h5file = h5py.File(h5filename, 'r')
data_coordx = h5file['_coord_x']
data_coordy = h5file['_coord_y']

nx = data_coordx.shape[0]
ny = data_coordy.shape[0]

print("From HDF5: (nx, ny) =", ny, nx)

#### GENERATE XDMF FILE ####

## general headers ##
xdmf = etree.Element("Xdmf", attrib={"Version": "2.0"})
domain = etree.SubElement(xdmf, "Domain")
grid = etree.SubElement(domain, "Grid", attrib={
    "Name": "grid1",
    "GridType": "Uniform"
})
topology = etree.SubElement(grid, "Topology", attrib={
    "TopologyType": "2DRectMesh",
    "Dimensions":   str(ny) + " " + str(nx)
})


## geometries ##
geometry = etree.SubElement(grid, "Geometry", attrib={"GeometryType": "VXVY"})
dataitem_coord_y = etree.SubElement(geometry, "DataItem", attrib={
    "Dimensions": str(ny),
    "NumberType": "Float",
    "Precision": "4",
    "Format": "HDF"
})
dataitem_coord_y.text = h5filename + ":/_coord_y"
dataitem_coord_x = etree.SubElement(geometry, "DataItem", attrib={
    "Dimensions": str(nx),
    "NumberType": "Float",
    "Precision": "4",
    "Format": "HDF"
})
dataitem_coord_x.text = h5filename + ":/_coord_x"


## datafields ##
for dset in h5file.keys():
    if dset[0] == '_':
        pass
    else:
        if len(h5file[dset].shape) != 2:
            print("Warning: dset " + dset + " is not 2D, skipping ...")
        else:
            print("Processing dset " + dset)

            attribute_data = etree.SubElement(grid, "Attribute", attrib={
                "Name": dset,
                "AttributeType": "Scalar",
                "Center": "Node"
            })
            dataitem_data = etree.SubElement(attribute_data, "DataItem", attrib={
                "Dimensions": str(ny) + " " + str(nx),
                "NumberType": "Float",
                "Precision": "4",
                "Format": "HDF"
            })
            dataitem_data.text = h5filename + ":/" + dset

h5file.close()

print("Writing to", xdmffilename)
tree = etree.ElementTree(xdmf)
tree.write(xdmffilename, pretty_print=True)

