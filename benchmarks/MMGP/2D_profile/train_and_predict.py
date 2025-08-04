import numpy as np
import copy
import pickle
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Muscat.IO.CGNSReader import ReadCGNS
from Muscat.FieldTransferKokkos.FieldTransfer import FieldTransferKokkos
from Muscat.Containers.Filters.FilterObjects import ElementFilter
from Muscat.FE.Fields.FEField import FEField
from Muscat.FE.FETools import PrepareFEComputation
from Muscat.Containers.Filters import FilterObjects as FO
from Muscat.IO import XdmfReader as XR
from scipy.sparse import identity

from model import GPyRegressor

from utils_2dprofile import POD
import os

plaid_location = # path to update to input plaid dataset



start=time.time()

# plaid_location_coarse = # path to update to plaid dataset containing a sample with the coarse common mesh
# reference_mesh_index=0
# reference_mesh_path = os.path.join(plaid_location_coarse, "dataset/samples/sample_00000000"+str(reference_mesh_index)+"/meshes/mesh_000000000.cgns")
# reference_mesh = ReadCGNS(fileName=reference_mesh_path)
reference_mesh = XR.ReadXdmf("coarse_common_mesh.xdmf")

reference_mesh_path = os.path.join(plaid_location, "dataset/samples/sample_00000000"+str(reference_mesh_index)+"/meshes/mesh_000000000.cgns")
reference_mesh_fin = ReadCGNS(fileName=reference_mesh_path)

space, numbering,_,_ = PrepareFEComputation(reference_mesh,numberOfComponents=1)
fefield = FEField("", mesh=reference_mesh, space=space, numbering=numbering[0])
op_ref_mesh=FieldTransferKokkos(inputField= fefield, targetPoints= reference_mesh_fin.nodes, method="Interp/Clamp" , elementFilter= ElementFilter(dimensionality=2) )[0]



node_fields=['Mach', 'Pressure', 'Velocity-x', 'Velocity-y']
n_train=300
nNodes_referenceMesh=reference_mesh_fin.nodes.shape[0]

flattend_data_morphing=np.zeros((n_train,2*reference_mesh.nodes.shape[0]))



# transfer data on reference mesh by pull back

data={}

for field_name in node_fields:
    data[field_name] = np.zeros((n_train,nNodes_referenceMesh))

morphing_displacement_fields=np.zeros((n_train,nNodes_referenceMesh,2))


for sample in range(n_train):
    print(sample)
    Tmesh_index=str(sample).zfill(3)

    file = open("displacement_field/displacement_field"+Tmesh_index+".pkl", 'rb')
    morphing_displacement_field_coarse = pickle.load(file)
    file.close()

    morphing_displacement_fields[sample] = op_ref_mesh.dot(morphing_displacement_field_coarse)

    temp_mesh=copy.deepcopy(reference_mesh_fin)

    temp_mesh.nodes +=  morphing_displacement_fields[sample]


    Tmesh_path = os.path.join(plaid_location, "dataset/samples/sample_00000000"+str(Tmesh_index)+"/meshes/mesh_000000000.cgns")
    Tmesh = ReadCGNS(fileName=Tmesh_path)

    space, numbering,_,_ = PrepareFEComputation(Tmesh,numberOfComponents=1)
    fefield = FEField("", mesh=Tmesh, space=space, numbering=numbering[0])
    OP=FieldTransferKokkos(inputField= fefield, targetPoints= temp_mesh.nodes, method="Interp/Clamp" , elementFilter= ElementFilter(dimensionality=2) )[0]

    for field_name in node_fields:
        data[field_name][sample] = OP.dot(Tmesh.nodeFields[field_name])


for sample in range(n_train):
    flattend_data_morphing[sample,:]=morphing_displacement_fields[sample,:,:].flatten()


#dimensionality reduction
elementFilter = FO.ElementFilter()
elementFilter.SetDimensionality(2)

correlationOperator1c = identity(reference_mesh_fin.nodes.shape[0])

correlationOperator2c = identity(2*reference_mesh.nodes.shape[0])

data_POD={}
reducedOrderBasis={}
generalizedCoordinates={}
eigenvalues={}

n_output= 40

for field_name in node_fields:
    reducedOrderBasis[field_name]  , generalizedCoordinates[field_name] , eigenvalues[field_name]  = POD(data[field_name],correlationOperator=correlationOperator1c,nmodes=n_output)
    print("Energy for field "+field_name+"= ", np.sum(eigenvalues[field_name] [:n_output])/np.sum(eigenvalues[field_name] [:]))


n_modes_input=18
reducedOrderBasis_u , generalizedCoordinates_u, eigenvalues_u = POD(data=flattend_data_morphing,correlationOperator=correlationOperator2c,nmodes=n_modes_input)
print("Energy for displacement fields = ", np.sum(eigenvalues_u[:n_modes_input])/np.sum(eigenvalues_u[:]))


# scaling inputs and outputs


scalerX= StandardScaler()
X=scalerX.fit_transform(generalizedCoordinates_u)

y_scalers = []
Y = []

for field_name in node_fields:
    y_scaler = MinMaxScaler()
    y_scalers.append(y_scaler)
    Y.append(y_scaler.fit_transform(generalizedCoordinates[field_name]))

# train GP

def train_single_output(X, y,num_restarts=7):
    gp = GPyRegressor(num_restarts=num_restarts)
    gp.fit(X, y)
    return gp



n_outputs = Y[0].shape[-1]

gp_models_fields=[]
print(f">> training field")

for i,field_name in enumerate(node_fields):

    print(f">> training {node_fields[i]}")

    model_one_field=[]
    for j in range(n_outputs):
        print(f"Coord. {j}")
        model=train_single_output(X,Y[i][:,j])
        model_one_field.append(model)

    gp_models_fields.append(model_one_field)




# predict

n_test=100

alpha=np.zeros((n_test,n_modes_input))
transfer_op_test=[]
for sample in range(n_test):


    print(sample)
    Tmesh_index=str(sample+n_train).zfill(3)

    file = open("displacement_field/displacement_field"+Tmesh_index+".pkl", 'rb')
    displacement_field = pickle.load(file)
    file.close()
    alpha[sample] = np.dot(reducedOrderBasis_u,correlationOperator2c.dot(displacement_field.flatten()))

    temp_mesh=copy.deepcopy(reference_mesh_fin)
    temp_mesh.nodes += op_ref_mesh.dot(displacement_field)


    Tmesh_path = os.path.join(plaid_location, "dataset/samples/sample_00000000"+str(Tmesh_index)+"/meshes/mesh_000000000.cgns")
    Tmesh = ReadCGNS(fileName=Tmesh_path)

    space, numbering,_,_ = PrepareFEComputation(temp_mesh,numberOfComponents=1)
    fefield = FEField("", mesh=temp_mesh, space=space, numbering=numbering[0])


    OP=FieldTransferKokkos(inputField= fefield, targetPoints= Tmesh.nodes, method="Interp/Clamp" , elementFilter= ElementFilter(dimensionality=2) )[0]
    transfer_op_test.append(OP)


input_test  = scalerX.transform(alpha)


y_pred_common = []



for i,field_name in enumerate(node_fields):
    output_dim = Y[i].shape[-1]
    y_pred_i = np.empty((n_test, output_dim))

    for j in range(output_dim):
        y_pred_i[:,j] = gp_models_fields[i][j].predict(input_test)[:,0]


    y_pred_i_inv = y_scalers[i].inverse_transform(y_pred_i)
    y_pred_common_i = np.dot(y_pred_i_inv, reducedOrderBasis[field_name])
    y_pred_common.append(y_pred_common_i)



prediction = []

for i in range(n_test):
    prediction.append({})


    for j,field_name in enumerate(node_fields):
        prediction[i][field_name] = transfer_op_test[i].dot(y_pred_common[j][i])

with open('prediction.pkl', 'wb') as file:
    pickle.dump(prediction, file)

