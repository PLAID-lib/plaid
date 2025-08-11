import numpy as np

from Muscat.FE.Spaces.FESpaces import LagrangeSpaceP0
from Muscat.FE.DofNumbering import ComputeDofNumbering
from Muscat.Containers import MeshInspectionTools as UMIP

from Muscat.Containers.Filters.FilterObjects import ElementFilter
import Muscat.Containers.MeshInspectionTools as UMIT
from Muscat.Containers.MeshModificationTools import  CleanLonelyNodes
from Muscat.FE.Fields.FEField import FEField
from Muscat.FE.FETools import PrepareFEComputation
from Muscat.FE.SymPhysics import MechPhysics
from Muscat.FE.SymWeakForm import GetField, GetScalarField,GetNormal
from Muscat.FE.UnstructuredFeaSym import UnstructuredFeaSym
from Muscat.FE.Fields.FieldTools import GetPointRepresentation
from Muscat.Helpers.Timer import Timer
from Muscat.Containers.MeshFieldOperations import GetFieldTransferOp
from Muscat.FE.FETools import  ComputeNormalsAtPoints


class MecaPhysics_ESM(MechPhysics):
    # Add weak formulations to the MecaPhysics class
    def __init__(self,dim=2   ,elasticModel ="isotropic"):
        super().__init__(dim)
        self.spaceDimension =dim
    def WeakDirichlet(self,alpha):

        u = self.primalUnknown
        ut = self.primalTest

        a = GetScalarField(alpha)
        return u.T* ut*a
    def WeakDirichletNormal(self,alpha):
        u = self.primalUnknown
        ut = self.primalTest
        a = GetScalarField(alpha)

        #Normal = GetNormal(self.spaceDimension )
        Normal = GetField("normal_nodes",2)


        return u.T*Normal * ut.T*Normal*a

    def vectDistanceFormulation(self):
        Normal = GetNormal(self.spaceDimension )
        ut = self.primalTest
        vectDist_symb=GetField("vectDist", size=2)

        return (vectDist_symb.T)*ut
        return (vectDist_symb.T*Normal)*ut.T*Normal

    def vectDistanceFormulation2(self):
        Normal = GetNormal(self.spaceDimension )
        ut = self.primalTest
        vectDist_symb=GetField("vectDist", size=2)
        return (vectDist_symb.T*ut)

    def Pressure_updated_normal(self, pressure):
        ut = self.primalTest

        p = GetScalarField(pressure)

        Normal = GetField("normal_nodes",2)

        return p * Normal.T * ut

def signedDistanceFunction(Tmesh, mesh,field_Tmesh,dim=1):

    #extract the boundary nodes of mesh.
    filter=ElementFilter(dimensionality=dim)
    boundary_ids=filter.GetNodesIndices(mesh=mesh)
    nNodes=mesh.GetNumberOfNodes()

    #calculate the distance. skinpos[i] is the projection of Mesh.nodes[i] on the boundary of Tmesh.
    opSkin, statusSkin, _  =  GetFieldTransferOp(inputField= field_Tmesh, targetPoints= mesh.nodes[boundary_ids], method="Interp/Clamp" , elementFilter= ElementFilter(dimensionality=1) , verbose=False)
    skinpos = opSkin.dot(Tmesh.nodes)

    signed_distance=np.zeros(nNodes)
    signed_distance[boundary_ids] = np.sqrt(np.sum((skinpos - mesh.nodes[boundary_ids])**2,axis=1))

    #calculate the sign. If statusBulk[i]==1, then Mesh.nodes[i] is inside Tmesh.
    _, statusBulk0, _ =  GetFieldTransferOp(inputField= field_Tmesh, targetPoints= mesh.nodes[boundary_ids], method="Interp/Clamp" , elementFilter= ElementFilter(dimensionality=2) , verbose=False)
    statusBulk=np.zeros(nNodes)
    statusBulk[boundary_ids]=statusBulk0[:,0]

    signed_distance[statusBulk[:]==1] *= -1

    return signed_distance


def ElasticProblem(mesh,extraFields,E=5.0 , nu=0.3 ,alpha=200,formulation="vect_distance",variable_E=False):

    problem = UnstructuredFeaSym()
    problem.fields = {f.name:f for f in extraFields}

    dim=2

    numberingP0 = ComputeDofNumbering(mesh,LagrangeSpaceP0,elementFilter=ElementFilter(dimensionality=dim) )
    volumes=abs(UMIP.GetVolumePerElement(mesh,elementFilter=ElementFilter(dimensionality=dim)))
    YoungModulusField=np.zeros(len(volumes))
    YoungModulusField[:]= ( 1+  1*(max(volumes)-min(volumes))/volumes[:]) * 0.5


    #YoungModulusField[:]=(1*(ratio[:]/max(ratio)-min(ratio))) * 0.02
    EField = FEField("E",mesh=mesh, space=LagrangeSpaceP0, numbering=numberingP0, data=YoungModulusField )

    problem.fields["E"]=EField


    normal_nodes=ComputeNormalsAtPoints(mesh)
    space, numbering,_,_ = PrepareFEComputation(mesh,numberOfComponents=1)
    a=normal_nodes[:,0].copy(order='C')
    b=normal_nodes[:,1].copy(order='C')
    normal_nodes_0=FEField("normal_nodes_0", mesh=mesh, space=space, numbering=numbering[0],data=a)
    normal_nodes_1=FEField("normal_nodes_1", mesh=mesh, space=space, numbering=numbering[0],data=b)

    problem.fields["normal_nodes_0"]=normal_nodes_0
    problem.fields["normal_nodes_1"]=normal_nodes_1

    P = 1

    # the main class

    # the mecanical problem
    mecaPhysics = MecaPhysics_ESM(dim=2)
    mecaPhysics.SetSpaceToLagrange(P=P)
    #mecaPhysics.integrationRule = "LagrangeP1Quadrature"

    #Left hand side operator
    if variable_E:
        mecaPhysics.AddBFormulation( ElementFilter(dimensionality=2) ,mecaPhysics.GetBulkFormulation('E',nu)  )
    else:
        mecaPhysics.AddBFormulation( ElementFilter(dimensionality=2) ,mecaPhysics.GetBulkFormulation(E,nu)  )

    mecaPhysics.AddBFormulation( ElementFilter( dimensionality=1) , data=mecaPhysics.WeakDirichletNormal(alpha) )
    mecaPhysics.AddBFormulation( ElementFilter( dimensionality=1,nTag="Ext_bound") , data=mecaPhysics.WeakDirichletNormal(100000000) )



    if formulation=="signed_distance":
        #term for the signed distance function formulation


        mecaPhysics.AddLFormulation( ElementFilter( dimensionality=1), mecaPhysics.GetPressureFormulation("signedDistance"))

        mecaPhysics.AddLFormulation( ElementFilter( dimensionality=1), data=mecaPhysics.vectDistanceFormulation())



    elif formulation=="vect_distance":
        #term for the vectorial distance function formulation
        mecaPhysics.AddLFormulation( ElementFilter( dimensionality=1), data=mecaPhysics.vectDistanceFormulation())
        mecaPhysics.AddLFormulation( ElementFilter( dimensionality=1), mecaPhysics.GetDistributedForceFormulation(["forceX", "forceY"]))
    else:
        raise ValueError("The formulation should be 'vect_distance' or 'signed_distance' ")

    #set the mesh, assemble the stiffnes matrix, solve and return the solution
    problem.physics.append(mecaPhysics)
    mesh.ConvertDataForNativeTreatment()
    problem.SetMesh(mesh)
    problem.ComputeDofNumbering()

    with Timer("Assembly "):
        k,f = problem.GetLinearProblem(computeK=True)
    problem.solver.SetAlgo("Direct")
    problem.ComputeConstraintsEquations()


    with Timer("Solve"):
        problem.Solve(k,f)

    problem.PushSolutionVectorToUnknownFields()


    return GetPointRepresentation(problem.unknownFields)


def VectorialDistance_Muscat_preprocessed(mesh,tags,Tmesh_partition={}):
    """ calculate for each node on the boundary of mesh with tag X, its closet point (projection) on the boundary of Tmesh with tag X.

    Args:
        mesh (Mesh): current mesh.
        tags (list, optional): list of strings (tag names), with tag name: string of the name of the name.
        Tmesh_partition(dict): dict of dict. Tmesh_partition[tag] contains :
            Tmesh_partition[tag]["mesh"]:  subsampled mesh of Tmesh containing the elements of tag.
            Tmesh_partition[tag]["FEField"]: FE Field on Tmesh_partition[tag]["mesh"].
    Returns:
        _numpy array_: _the vector distance function_
    """
    vectDist = np.zeros((len(mesh.nodes),2))

    for tag in tags:

        #extract the IDs of the nodes on mesh with the current tag
        mesh_Filter= ElementFilter(nTag =tag)
        mesh_ids=mesh_Filter.GetNodesIndices(mesh=mesh)

        #calculate the closet point on Tmesh_tag


        Tmesh_partition[tag]["TransferOperator"].SetTargetPoints( mesh.nodes[mesh_ids])
        Tmesh_partition[tag]["TransferOperator"].Compute()

        operator = Tmesh_partition[tag]["TransferOperator"].GetOperator()

        skinpos = operator.dot(Tmesh_partition[tag]["mesh"].nodes)

        vectDist[mesh_ids] = (skinpos - mesh.nodes[mesh_ids])

    return vectDist


def signedDistannce_Function_kokkos(Tmesh, mesh,field_Tmesh,dim=1):
        #extract the boundary nodes of mesh.
    filter=ElementFilter(dimensionality=dim,nTag=["Airfoil"])
    boundary_ids=filter.GetNodesIndices(mesh=mesh)



    TagMesh=UMIT.ExtractElementsByElementFilter(Tmesh, filter)
    CleanLonelyNodes(TagMesh)
    Tspace, Tnumberings,_,_ = PrepareFEComputation(TagMesh,numberOfComponents=1,elementFilter= ElementFilter(dimensionality=1,nTag=["Airfoil"]))
    field_Tmesh_boundary = FEField("", mesh=TagMesh, space=Tspace, numbering=Tnumberings[0])


    nNodes=mesh.GetNumberOfNodes()

    #calculate the distance. skinpos[i] is the projection of Mesh.nodes[i] on the boundary of Tmesh.
    opSkin  =  GetFieldTransferOp(inputField= field_Tmesh_boundary, targetPoints= mesh.nodes[boundary_ids], method="Interp/Clamp" , elementFilter= ElementFilter(dimensionality=1,nTag=["Airfoil"]) )[0]
    skinpos = opSkin.dot(TagMesh.nodes)

    signed_distance=np.zeros(nNodes)
    signed_distance[boundary_ids] = np.sqrt(np.sum((skinpos - mesh.nodes[boundary_ids])**2,axis=1))

    #calculate the sign. If statusBulk[i]==1, then Mesh.nodes[i] is inside Tmesh.
    statusBulk0=  GetFieldTransferOp(inputField= field_Tmesh, targetPoints= mesh.nodes[boundary_ids], method="Interp/Clamp" , elementFilter= ElementFilter(dimensionality=2) )[1]
    statusBulk=np.zeros(nNodes)

    statusBulk[boundary_ids]=statusBulk0[:,0]

    signed_distance[statusBulk[:]==1] *= -1

    return signed_distance



def POD(data,correlationOperator,nmodes):
    Nsamples=data.shape[0]
    matVecProducts_x=correlationOperator.dot(data[:,:].T)
    correlation_matrix_morphing=np.dot(matVecProducts_x.T ,data[:,:].T)
    eigenvalues_ux, eigenvectors_ux = np.linalg.eigh(correlation_matrix_morphing)
    idx = eigenvalues_ux.argsort()[::-1]
    eigenvalues_ux = eigenvalues_ux[idx]
    eigenvectors_ux = eigenvectors_ux[:, idx]


    changeOfBasisMatrix_x = np.zeros((nmodes,Nsamples))
    for j in range(nmodes):
        changeOfBasisMatrix_x[j,:] = eigenvectors_ux[:,j]/np.sqrt(eigenvalues_ux[j])

    reducedOrderBasis_x = np.dot(changeOfBasisMatrix_x,data[:,:])
    generalizedCoordinates_x= np.dot(reducedOrderBasis_x, matVecProducts_x).T

    return reducedOrderBasis_x , generalizedCoordinates_x, eigenvalues_ux