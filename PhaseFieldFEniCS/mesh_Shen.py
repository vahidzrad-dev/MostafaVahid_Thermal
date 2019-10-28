# FEnics code  Gmsh
# Mostafa Mollaali
# Vahid Ziaei-Rad


from fenics import *
from dolfin import *
from mshr import *
import sympy, sys, math, os, subprocess, shutil
from subprocess import call
from dolfin_utils.meshconvert import meshconvert


#=======================================================================================
# Input date
#=======================================================================================
hsize=5.0e-3

#=======================================================================================
# Geometry and mesh generation
#=======================================================================================
meshname="mesh"

# Generate a XDMF/HDF5 based mesh from a Gmsh string
geofile = \
        """
            lc = DefineNumber[ %g, Name "Parameters/lc" ];
            H = 500.0e-3;
            L = 500.0e-3;
            t = 10.0e-3;
            
            

            Point(1) = {0, 0, 0, lc};
            Point(2) = {L, 0, 0, lc};
            
            Point(3) = {L, H, 0, lc};
            Point(4) = {0, H, 0, lc};
            

            Point(5) = {0, 0, t, lc};
            Point(6) = {L, 0, t, lc};
            
            Point(7) = {L, H, t, lc};
            Point(8) = {0, H, t, lc};
            
            
            Line(1) = {1, 2};
            Line(2) = {2, 3};
            Line(3) = {3, 4};
            Line(4) = {4, 1};
            
            
            Line(5) = {5, 6};
            Line(6) = {6, 7};
            Line(7) = {7, 8};
            Line(8) = {8, 5};
            
            
            Line(9) = {1, 5};
            Line(10) = {2, 6};
            Line(11) = {3, 7};
            Line(12) = {4, 8};
            
            Line Loop(1) = {1, 2, 3, 4};
            Line Loop(2) = {5, 6, 7, 8};
            Line Loop(3) = {1, 2, 6, 5};
            Line Loop(4) = {2, 3, 7, 6};
            Line Loop(5) = {3, 4, 8, 7};
            Line Loop(6) = {4, 1, 5, 8};
            
            Plane Surface(1) = {1};
            Plane Surface(2) = {2};
            Plane Surface(3) = {3};
            Plane Surface(4) = {4};
            Plane Surface(5) = {5};
            Plane Surface(6) = {6};
            
            Compound Volume(2) = {1};
            
            Physical Surface(2) = {2};
            Physical Surface(2) = {2};

"""%(hsize)


subdir = "meshes_Shen/"
_mesh  = Mesh() #creat empty mesh object


if not os.path.isfile(subdir + meshname + ".xdmf"):
        if MPI.comm_world.rank == 0:
            # Create temporary .geo file defining the mesh
            if os.path.isdir(subdir) == False:
                os.mkdir(subdir)
            fgeo = open(subdir + meshname + ".geo", "w")
            fgeo.writelines(geofile)
            fgeo.close()
            # Calling gmsh and dolfin-convert to generate the .xml mesh (as well as a MeshFunction file)
            try:
                    subprocess.call(["gmsh", "-3", "-o", subdir + meshname + ".msh", subdir + meshname + ".geo"])
            except OSError:
                    print("-----------------------------------------------------------------------------")
                    print(" Error: unable to generate the mesh using gmsh")
                    print(" Make sure that you have gmsh installed and have added it to your system PATH")
                    print("-----------------------------------------------------------------------------")


            meshconvert.convert2xml(subdir + meshname + ".msh", subdir + meshname + ".xml", "gmsh")

            # Convert to XDMF
            MPI.barrier(MPI.comm_world)
            mesh = Mesh(subdir + meshname + ".xml")
            XDMF = XDMFFile(MPI.comm_world, subdir + meshname + ".xdmf")
            XDMF.write(mesh)
            XDMF.read(_mesh)
        
        if os.path.isfile(subdir + meshname + "_physical_region.xml") and os.path.isfile(subdir + meshname + "_facet_region.xml"):
            if MPI.comm_world.rank == 0:
                    mesh = Mesh(subdir + meshname + ".xml")
                    subdomains = MeshFunction("size_t", mesh, subdir + meshname + "_physical_region.xml")
                    boundaries = MeshFunction("size_t", mesh, subdir + meshname + "_facet_region.xml")
                    HDF5 = HDF5File(MPI.comm_world, subdir + meshname + "_physical_facet.h5", "w")
                    HDF5.write(mesh, "/mesh")
                    HDF5.write(subdomains, "/subdomains")
                    HDF5.write(boundaries, "/boundaries")
                    print("Finish writting physical_facet to HDF5")

        print("Mesh completed")

    # Read the mesh if existing
else:
        XDMF = XDMFFile(MPI.comm_world, subdir + meshname + ".xdmf")
        
        XDMF.read(_mesh)
