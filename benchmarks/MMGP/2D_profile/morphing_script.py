
import pickle
import copy
import time
import sys , os

from Muscat.FE.Fields.FEField import FEField
from Muscat.FE.FETools import PrepareFEComputation
from Muscat.Containers.Filters.FilterObjects import ElementFilter
from Muscat.Containers.NativeTransfer import NativeTransfer
from Muscat.IO.CGNSReader import ReadCGNS
from Muscat.IO import XdmfReader as XR
import Muscat.Containers.MeshInspectionTools as UMIT
from Muscat.Containers.MeshModificationTools import  CleanLonelyNodes , ComputeSkin

from utils_2dprofile import  ElasticProblem  , VectorialDistance_Muscat_preprocessed,signedDistannce_Function_kokkos


plaid_location = "/path/to/plaid/" # path to update to input plaid dataset


def MatchTwoGeometries(mesh, Tmesh,TmeshIndex=250,max_iteration=200, tolerance= 1*10**(-3) ,YoungModulus=0.1 , nu=0.3 , alpha=200, gamma=5,beta=0,formulation="vect_distance",tags=[]) :

    space, numbering,_,_ = PrepareFEComputation(mesh,numberOfComponents=1)

    Tspace, Tnumberings,_,_ = PrepareFEComputation(Tmesh,numberOfComponents=1)
    field_Tmesh = FEField("", mesh=Tmesh, space=Tspace, numbering=Tnumberings[0])

    begin=time.time()


    Tmesh_partition={}
    for tag in tags:
        Tmesh_partition[tag]={}
        mesh_Filter= ElementFilter(nTag =tag)
        Tmesh_tag=UMIT.ExtractElementsByElementFilter(Tmesh, mesh_Filter)
        CleanLonelyNodes(Tmesh_tag)
        Tspace, Tnumberings,_,_ = PrepareFEComputation(Tmesh_tag,numberOfComponents=1)
        FE_Tmesh_tag = FEField("", mesh=Tmesh_tag, space=Tspace, numbering=Tnumberings[0])
        Tmesh_partition[tag]["mesh"]=Tmesh_tag
        Tmesh_partition[tag]["FEField"]=FE_Tmesh_tag

        nt = NativeTransfer()
        nt.SetVerbose(False)
        nt.SetTransferMethod("Interp/Clamp")
        nt.SetSourceFEField(FE_Tmesh_tag, elementFilter=None)
        Tmesh_partition[tag]["TransferOperator"]=nt


    space, numbering,_,_ = PrepareFEComputation(mesh,numberOfComponents=1)

    Tspace, Tnumberings,_,_ = PrepareFEComputation(Tmesh,numberOfComponents=1)
    field_Tmesh = FEField("", mesh=Tmesh, space=Tspace, numbering=Tnumberings[0])

    begin=time.time()


    sol = mesh.nodes*0


    dist   = signedDistannce_Function_kokkos(Tmesh, mesh,field_Tmesh=field_Tmesh)

    d=    max(abs(dist))

    begin=time.time()
    i=0


    variable_E=True
    tranfer_time=0

    while d > tolerance:

        print(f"Iteration {i}")
        print(d)
        if i%40 ==0 and i<340:
            #print(i)
            variable_E =True
            alpha=alpha*2
            gamma = gamma*2

        vectDist=VectorialDistance_Muscat_preprocessed(mesh,tags=tags,Tmesh_partition=Tmesh_partition)
        signedDistance   = signedDistannce_Function_kokkos(Tmesh, mesh,field_Tmesh=field_Tmesh)
        data_signedDistance= signedDistance*  -0.25

        if i ==600:
            #print(i)
            variable_E =False
            alpha=50

            vectDist = vectDist*0.0
            data_signedDistance= signedDistance*  2

            gamma = gamma*0.007


        d=    max(abs(signedDistance))
        extraFields = [FEField("vectDist_0",mesh=mesh, space=space, numbering=numbering[0], data=  1*vectDist[:,0]),
                       FEField("vectDist_1",mesh=mesh, space=space, numbering=numbering[0], data=  1*vectDist[:,1]),
                       FEField("signedDistance",mesh=Tmesh, space=space, numbering=numbering[0], data=  data_signedDistance)]


        sol =  ElasticProblem(mesh, extraFields,YoungModulus,nu,alpha,formulation=formulation,variable_E=variable_E)



        mesh.nodes += sol*gamma
        if i==max_iteration:
            time_per_sample=time.time()-begin
            print("Time=",time_per_sample)

            break
        i+=1

    time_per_sample=time.time()-begin

    print("number of iteration= ", i)
    print("Time=",time_per_sample)
    print("tranfer time = ", tranfer_time)

    return mesh



tags=["Airfoil"]
sample=int(sys.argv[1])
print("sample = ", sample)

Tmesh_index=str(sample).zfill(3)

Tmesh_path = os.path.join(plaid_location, "dataset/samples/sample_000000"+str(Tmesh_index)+"/meshes/mesh_000000000.cgns")

# plaid_location_coarse = # path to update to plaid dataset containing a sample with the coarse common mesh
# reference_mesh_index=0
# reference_mesh_path = os.path.join(plaid_location_coarse, "dataset/samples/sample_00000000"+str(reference_mesh_index)+"/meshes/mesh_000000000.cgns")
# reference_mesh = ReadCGNS(fileName=reference_mesh_path)
reference_mesh = XR.ReadXdmf("coarse_common_mesh.xdmf")
Tmesh = ReadCGNS(fileName=Tmesh_path)

ComputeSkin(reference_mesh,inPlace=True)
ComputeSkin(Tmesh,inPlace=True)

mesh =MatchTwoGeometries(copy.deepcopy(reference_mesh), Tmesh ,max_iteration=700 ,tolerance= 5*10**(-5) ,YoungModulus=1, nu=0.3 , alpha=100, gamma=15,beta=0,
                                                formulation="signed_distance",tags=tags)


displacement_field=mesh.nodes-reference_mesh.nodes


folder_path = "displacement_field"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open(folder_path+"/displacement_field"+Tmesh_index+".pkl", 'wb') as file:
    pickle.dump(displacement_field, file)