??
?&?%
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8ٻ
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*%
shared_nameembedding/embeddings

(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings* 
_output_shapes
:
??@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
string_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_13*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*,
shared_nameAdam/embedding/embeddings/m
?
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m* 
_output_shapes
:
??@*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:@@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*,
shared_nameAdam/embedding/embeddings/v
?
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v* 
_output_shapes
:
??@*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:@@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_17928

NoOpNoOp^PartitionedCall
?
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_index_table*
Tkeys0*
Tvalues0	*,
_class"
 loc:@string_lookup_index_table*
_output_shapes

::
?&
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?%B?% B?%
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
=
state_variables
_index_lookup_layer
	keras_api
b

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
?
$iter

%beta_1

&beta_2
	'decay
(learning_ratemPmQmRmSmTvUvVvWvXvY
 
#
1
2
3
4
5
#
0
1
2
3
4
?
regularization_losses

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
-non_trainable_variables
	trainable_variables
 
 
0
.state_variables

/_table
0	keras_api
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
regularization_losses

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
5non_trainable_variables
trainable_variables
 
 
 
?
regularization_losses

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
:non_trainable_variables
trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
?non_trainable_variables
trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
 regularization_losses

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
!	variables
Dnon_trainable_variables
"trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
#
0
1
2
3
4

E0
F1
 
 
 
 
LJ
tableAlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Gtotal
	Hcount
I	variables
J	keras_api
D
	Ktotal
	Lcount
M
_fn_kwargs
N	variables
O	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

G0
H1

I	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1

N	variables
??
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_vectorize_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_vectorize_inputstring_lookup_index_tableConstembedding/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_17542
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpHstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst_1*'
Tin 
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_18030
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratestring_lookup_index_tabletotalcounttotal_1count_1Adam/embedding/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/embedding/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_18115??
?
T
8__inference_global_average_pooling1d_layer_call_fn_17830

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_171392
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_17765

inputsQ
Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleR
Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	6
2embedding_embedding_lookup_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?)embedding/embedding_lookup/ReadVariableOp?@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2?
vectorize/SqueezeSqueezeinputs*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
vectorize/Squeeze{
vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
vectorize/StringSplit/Const?
#vectorize/StringSplit/StringSplitV2StringSplitV2vectorize/Squeeze:output:0$vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2%
#vectorize/StringSplit/StringSplitV2?
)vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)vectorize/StringSplit/strided_slice/stack?
+vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+vectorize/StringSplit/strided_slice/stack_1?
+vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+vectorize/StringSplit/strided_slice/stack_2?
#vectorize/StringSplit/strided_sliceStridedSlice-vectorize/StringSplit/StringSplitV2:indices:02vectorize/StringSplit/strided_slice/stack:output:04vectorize/StringSplit/strided_slice/stack_1:output:04vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2%
#vectorize/StringSplit/strided_slice?
+vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+vectorize/StringSplit/strided_slice_1/stack?
-vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_1?
-vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_2?
%vectorize/StringSplit/strided_slice_1StridedSlice+vectorize/StringSplit/StringSplitV2:shape:04vectorize/StringSplit/strided_slice_1/stack:output:06vectorize/StringSplit/strided_slice_1/stack_1:output:06vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2'
%vectorize/StringSplit/strided_slice_1?
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast,vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2N
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast.vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapePvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProd_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreater^vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0cvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2]vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulYvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2[
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsum`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2Y
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle,vectorize/StringSplit/StringSplitV2:values:0Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2t
)vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2+
)vectorize/string_lookup/assert_equal/NoOp?
 vectorize/string_lookup/IdentityIdentityIvectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2"
 vectorize/string_lookup/Identity?
"vectorize/string_lookup/Identity_1IdentityWvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????2$
"vectorize/string_lookup/Identity_1?
&vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2(
&vectorize/RaggedToTensor/default_value?
vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2 
vectorize/RaggedToTensor/Const?
-vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor'vectorize/RaggedToTensor/Const:output:0)vectorize/string_lookup/Identity:output:0/vectorize/RaggedToTensor/default_value:output:0+vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2/
-vectorize/RaggedToTensor/RaggedTensorToTensor?
vectorize/ShapeShape6vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
vectorize/Shape?
vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
vectorize/strided_slice/stack?
vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_1?
vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_2?
vectorize/strided_sliceStridedSlicevectorize/Shape:output:0&vectorize/strided_slice/stack:output:0(vectorize/strided_slice/stack_1:output:0(vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vectorize/strided_slicee
vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/sub/x?
vectorize/subSubvectorize/sub/x:output:0 vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
vectorize/subg
vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/Less/y?
vectorize/LessLess vectorize/strided_slice:output:0vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
vectorize/Less?
vectorize/condStatelessIfvectorize/Less:z:0vectorize/sub:z:06vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
else_branchR
vectorize_cond_false_17723*/
output_shapes
:??????????????????*,
then_branchR
vectorize_cond_true_177222
vectorize/cond?
vectorize/cond/IdentityIdentityvectorize/cond:output:0*
T0	*(
_output_shapes
:??????????2
vectorize/cond/Identity?
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
??@*
dtype02+
)embedding/embedding_lookup/ReadVariableOp?
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axis?
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0 vectorize/cond/Identity:output:0(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*,
_output_shapes
:??????????@2
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*,
_output_shapes
:??????????@2%
#embedding/embedding_lookup/Identity?
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices?
global_average_pooling1d/MeanMean,embedding/embedding_lookup/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
global_average_pooling1d/Mean?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
IdentityIdentitydense_1/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^embedding/embedding_lookup/ReadVariableOpA^vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)embedding/embedding_lookup/ReadVariableOp)embedding/embedding_lookup/ReadVariableOp2?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?

?
D__inference_embedding_layer_call_and_return_conditional_losses_17812

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
??@*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*,
_output_shapes
:??????????@2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_output_shapes
:??????????@2
embedding_lookup/Identity?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_17825

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?n
?
!__inference__traced_restore_18115
file_prefix)
%assignvariableop_embedding_embeddings#
assignvariableop_1_dense_kernel!
assignvariableop_2_dense_bias%
!assignvariableop_3_dense_1_kernel#
assignvariableop_4_dense_1_bias 
assignvariableop_5_adam_iter"
assignvariableop_6_adam_beta_1"
assignvariableop_7_adam_beta_2!
assignvariableop_8_adam_decay)
%assignvariableop_9_adam_learning_rateY
Ustring_lookup_index_table_table_restore_lookuptableimportv2_string_lookup_index_table
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_13
/assignvariableop_14_adam_embedding_embeddings_m+
'assignvariableop_15_adam_dense_kernel_m)
%assignvariableop_16_adam_dense_bias_m-
)assignvariableop_17_adam_dense_1_kernel_m+
'assignvariableop_18_adam_dense_1_bias_m3
/assignvariableop_19_adam_embedding_embeddings_v+
'assignvariableop_20_adam_dense_kernel_v)
%assignvariableop_21_adam_dense_bias_v-
)assignvariableop_22_adam_dense_1_kernel_v+
'assignvariableop_23_adam_dense_1_bias_v
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?;string_lookup_index_table_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9?
;string_lookup_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ustring_lookup_index_table_table_restore_lookuptableimportv2_string_lookup_index_tableRestoreV2:tensors:10RestoreV2:tensors:11*	
Tin0*

Tout0	*,
_class"
 loc:@string_lookup_index_table*
_output_shapes
 2=
;string_lookup_index_table_table_restore/LookupTableImportV2n
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_adam_embedding_embeddings_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_adam_embedding_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_dense_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_1_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_1_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp<^string_lookup_index_table_table_restore/LookupTableImportV2"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24?
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9<^string_lookup_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92z
;string_lookup_index_table_table_restore/LookupTableImportV2;string_lookup_index_table_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:2.
,
_class"
 loc:@string_lookup_index_table
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_17496

inputsQ
Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleR
Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
embedding_17481
dense_17485
dense_17487
dense_1_17490
dense_1_17492
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2?
vectorize/SqueezeSqueezeinputs*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
vectorize/Squeeze{
vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
vectorize/StringSplit/Const?
#vectorize/StringSplit/StringSplitV2StringSplitV2vectorize/Squeeze:output:0$vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2%
#vectorize/StringSplit/StringSplitV2?
)vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)vectorize/StringSplit/strided_slice/stack?
+vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+vectorize/StringSplit/strided_slice/stack_1?
+vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+vectorize/StringSplit/strided_slice/stack_2?
#vectorize/StringSplit/strided_sliceStridedSlice-vectorize/StringSplit/StringSplitV2:indices:02vectorize/StringSplit/strided_slice/stack:output:04vectorize/StringSplit/strided_slice/stack_1:output:04vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2%
#vectorize/StringSplit/strided_slice?
+vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+vectorize/StringSplit/strided_slice_1/stack?
-vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_1?
-vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_2?
%vectorize/StringSplit/strided_slice_1StridedSlice+vectorize/StringSplit/StringSplitV2:shape:04vectorize/StringSplit/strided_slice_1/stack:output:06vectorize/StringSplit/strided_slice_1/stack_1:output:06vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2'
%vectorize/StringSplit/strided_slice_1?
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast,vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2N
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast.vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapePvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProd_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreater^vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0cvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2]vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulYvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2[
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsum`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2Y
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle,vectorize/StringSplit/StringSplitV2:values:0Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2t
)vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2+
)vectorize/string_lookup/assert_equal/NoOp?
 vectorize/string_lookup/IdentityIdentityIvectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2"
 vectorize/string_lookup/Identity?
"vectorize/string_lookup/Identity_1IdentityWvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????2$
"vectorize/string_lookup/Identity_1?
&vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2(
&vectorize/RaggedToTensor/default_value?
vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2 
vectorize/RaggedToTensor/Const?
-vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor'vectorize/RaggedToTensor/Const:output:0)vectorize/string_lookup/Identity:output:0/vectorize/RaggedToTensor/default_value:output:0+vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2/
-vectorize/RaggedToTensor/RaggedTensorToTensor?
vectorize/ShapeShape6vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
vectorize/Shape?
vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
vectorize/strided_slice/stack?
vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_1?
vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_2?
vectorize/strided_sliceStridedSlicevectorize/Shape:output:0&vectorize/strided_slice/stack:output:0(vectorize/strided_slice/stack_1:output:0(vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vectorize/strided_slicee
vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/sub/x?
vectorize/subSubvectorize/sub/x:output:0 vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
vectorize/subg
vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/Less/y?
vectorize/LessLess vectorize/strided_slice:output:0vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
vectorize/Less?
vectorize/condStatelessIfvectorize/Less:z:0vectorize/sub:z:06vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
else_branchR
vectorize_cond_false_17461*/
output_shapes
:??????????????????*,
then_branchR
vectorize_cond_true_174602
vectorize/cond?
vectorize/cond/IdentityIdentityvectorize/cond:output:0*
T0	*(
_output_shapes
:??????????2
vectorize/cond/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCall vectorize/cond/Identity:output:0embedding_17481*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_171222#
!embedding/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_171392*
(global_average_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_17485dense_17487*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_171572
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_17490dense_1_17492*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_171842!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCallA^vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_17872

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
T
8__inference_global_average_pooling1d_layer_call_fn_17841

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_170332
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?<
?
__inference__traced_save_18030
file_prefix3
/savev2_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopS
Osavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2U
Qsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const_1

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopOsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2Qsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *)
dtypes
2		2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??@:@@:@:@:: : : : : ::: : : : :
??@:@@:@:@::
??@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::&"
 
_output_shapes
:
??@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
?
?
*__inference_sequential_layer_call_fn_17513
vectorize_input
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallvectorize_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_174962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namevectorize_input:

_output_shapes
: 
?
I
__inference__creator_17886
identity??string_lookup_index_table?
string_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_13*
value_dtype0	2
string_lookup_index_table?
IdentityIdentity(string_lookup_index_table:table_handle:0^string_lookup_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 26
string_lookup_index_tablestring_lookup_index_table
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_17033

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
vectorize_cond_true_17092/
+vectorize_cond_pad_paddings_1_vectorize_subD
@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2!
vectorize/cond/Pad/paddings/1/0?
vectorize/cond/Pad/paddings/1Pack(vectorize/cond/Pad/paddings/1/0:output:0+vectorize_cond_pad_paddings_1_vectorize_sub*
N*
T0*
_output_shapes
:2
vectorize/cond/Pad/paddings/1?
vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
vectorize/cond/Pad/paddings/0_1?
vectorize/cond/Pad/paddingsPack(vectorize/cond/Pad/paddings/0_1:output:0&vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2
vectorize/cond/Pad/paddings?
vectorize/cond/PadPad@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor$vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Pad?
vectorize/cond/IdentityIdentityvectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_17139

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
vectorize_cond_false_17351
vectorize_cond_placeholderN
Jvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
"vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"vectorize/cond/strided_slice/stack?
$vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2&
$vectorize/cond/strided_slice/stack_1?
$vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$vectorize/cond/strided_slice/stack_2?
vectorize/cond/strided_sliceStridedSliceJvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor+vectorize/cond/strided_slice/stack:output:0-vectorize/cond/strided_slice/stack_1:output:0-vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2
vectorize/cond/strided_slice?
vectorize/cond/IdentityIdentity%vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
__inference_save_fn_17915
checkpoint_keyY
Ustring_lookup_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2?
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Ustring_lookup_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2J
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityOstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:keys:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
:2

Identity_2?

Identity_3Identity	add_1:z:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityQstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:values:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_17157

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_17784

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_173862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
__inference_restore_fn_17923
restored_tensors_0
restored_tensors_1	L
Hstring_lookup_index_table_table_restore_lookuptableimportv2_table_handle
identity??;string_lookup_index_table_table_restore/LookupTableImportV2?
;string_lookup_index_table_table_restore/LookupTableImportV2LookupTableImportV2Hstring_lookup_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2=
;string_lookup_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0<^string_lookup_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:::2z
;string_lookup_index_table_table_restore/LookupTableImportV2;string_lookup_index_table_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
?
vectorize_cond_false_17093
vectorize_cond_placeholderN
Jvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
"vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"vectorize/cond/strided_slice/stack?
$vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2&
$vectorize/cond/strided_slice/stack_1?
$vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$vectorize/cond/strided_slice/stack_2?
vectorize/cond/strided_sliceStridedSliceJvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor+vectorize/cond/strided_slice/stack:output:0-vectorize/cond/strided_slice/stack_1:output:0-vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2
vectorize/cond/strided_slice?
vectorize/cond/IdentityIdentity%vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
|
'__inference_dense_1_layer_call_fn_17881

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_171842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_17184

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_17386

inputsQ
Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleR
Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
embedding_17371
dense_17375
dense_17377
dense_1_17380
dense_1_17382
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2?
vectorize/SqueezeSqueezeinputs*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
vectorize/Squeeze{
vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
vectorize/StringSplit/Const?
#vectorize/StringSplit/StringSplitV2StringSplitV2vectorize/Squeeze:output:0$vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2%
#vectorize/StringSplit/StringSplitV2?
)vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)vectorize/StringSplit/strided_slice/stack?
+vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+vectorize/StringSplit/strided_slice/stack_1?
+vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+vectorize/StringSplit/strided_slice/stack_2?
#vectorize/StringSplit/strided_sliceStridedSlice-vectorize/StringSplit/StringSplitV2:indices:02vectorize/StringSplit/strided_slice/stack:output:04vectorize/StringSplit/strided_slice/stack_1:output:04vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2%
#vectorize/StringSplit/strided_slice?
+vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+vectorize/StringSplit/strided_slice_1/stack?
-vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_1?
-vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_2?
%vectorize/StringSplit/strided_slice_1StridedSlice+vectorize/StringSplit/StringSplitV2:shape:04vectorize/StringSplit/strided_slice_1/stack:output:06vectorize/StringSplit/strided_slice_1/stack_1:output:06vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2'
%vectorize/StringSplit/strided_slice_1?
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast,vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2N
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast.vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapePvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProd_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreater^vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0cvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2]vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulYvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2[
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsum`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2Y
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle,vectorize/StringSplit/StringSplitV2:values:0Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2t
)vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2+
)vectorize/string_lookup/assert_equal/NoOp?
 vectorize/string_lookup/IdentityIdentityIvectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2"
 vectorize/string_lookup/Identity?
"vectorize/string_lookup/Identity_1IdentityWvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????2$
"vectorize/string_lookup/Identity_1?
&vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2(
&vectorize/RaggedToTensor/default_value?
vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2 
vectorize/RaggedToTensor/Const?
-vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor'vectorize/RaggedToTensor/Const:output:0)vectorize/string_lookup/Identity:output:0/vectorize/RaggedToTensor/default_value:output:0+vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2/
-vectorize/RaggedToTensor/RaggedTensorToTensor?
vectorize/ShapeShape6vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
vectorize/Shape?
vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
vectorize/strided_slice/stack?
vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_1?
vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_2?
vectorize/strided_sliceStridedSlicevectorize/Shape:output:0&vectorize/strided_slice/stack:output:0(vectorize/strided_slice/stack_1:output:0(vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vectorize/strided_slicee
vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/sub/x?
vectorize/subSubvectorize/sub/x:output:0 vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
vectorize/subg
vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/Less/y?
vectorize/LessLess vectorize/strided_slice:output:0vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
vectorize/Less?
vectorize/condStatelessIfvectorize/Less:z:0vectorize/sub:z:06vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
else_branchR
vectorize_cond_false_17351*/
output_shapes
:??????????????????*,
then_branchR
vectorize_cond_true_173502
vectorize/cond?
vectorize/cond/IdentityIdentityvectorize/cond:output:0*
T0	*(
_output_shapes
:??????????2
vectorize/cond/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCall vectorize/cond/Identity:output:0embedding_17371*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_171222#
!embedding/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_171392*
(global_average_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_17375dense_17377*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_171572
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_17380dense_1_17382*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_171842!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCallA^vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
vectorize_cond_true_17256/
+vectorize_cond_pad_paddings_1_vectorize_subD
@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2!
vectorize/cond/Pad/paddings/1/0?
vectorize/cond/Pad/paddings/1Pack(vectorize/cond/Pad/paddings/1/0:output:0+vectorize_cond_pad_paddings_1_vectorize_sub*
N*
T0*
_output_shapes
:2
vectorize/cond/Pad/paddings/1?
vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
vectorize/cond/Pad/paddings/0_1?
vectorize/cond/Pad/paddingsPack(vectorize/cond/Pad/paddings/0_1:output:0&vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2
vectorize/cond/Pad/paddings?
vectorize/cond/PadPad@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor$vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Pad?
vectorize/cond/IdentityIdentityvectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
*__inference_sequential_layer_call_fn_17803

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_174962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
*
__inference_<lambda>_17928
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
??
?
 __inference__wrapped_model_17017
vectorize_input\
Xsequential_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle]
Ysequential_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	A
=sequential_embedding_embedding_lookup_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?4sequential/embedding/embedding_lookup/ReadVariableOp?Ksequential/vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2?
sequential/vectorize/SqueezeSqueezevectorize_input*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
sequential/vectorize/Squeeze?
&sequential/vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2(
&sequential/vectorize/StringSplit/Const?
.sequential/vectorize/StringSplit/StringSplitV2StringSplitV2%sequential/vectorize/Squeeze:output:0/sequential/vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:20
.sequential/vectorize/StringSplit/StringSplitV2?
4sequential/vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4sequential/vectorize/StringSplit/strided_slice/stack?
6sequential/vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6sequential/vectorize/StringSplit/strided_slice/stack_1?
6sequential/vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6sequential/vectorize/StringSplit/strided_slice/stack_2?
.sequential/vectorize/StringSplit/strided_sliceStridedSlice8sequential/vectorize/StringSplit/StringSplitV2:indices:0=sequential/vectorize/StringSplit/strided_slice/stack:output:0?sequential/vectorize/StringSplit/strided_slice/stack_1:output:0?sequential/vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask20
.sequential/vectorize/StringSplit/strided_slice?
6sequential/vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential/vectorize/StringSplit/strided_slice_1/stack?
8sequential/vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential/vectorize/StringSplit/strided_slice_1/stack_1?
8sequential/vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential/vectorize/StringSplit/strided_slice_1/stack_2?
0sequential/vectorize/StringSplit/strided_slice_1StridedSlice6sequential/vectorize/StringSplit/StringSplitV2:shape:0?sequential/vectorize/StringSplit/strided_slice_1/stack:output:0Asequential/vectorize/StringSplit/strided_slice_1/stack_1:output:0Asequential/vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask22
0sequential/vectorize/StringSplit/strided_slice_1?
Wsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7sequential/vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2Y
Wsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Ysequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9sequential/vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2[
Ysequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
asequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2c
asequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
asequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2c
asequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
`sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2b
`sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
esequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2g
esequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterisequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0nsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2e
csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
`sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2b
`sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2e
csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
_sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0lsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2a
_sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
asequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2c
asequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
_sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2hsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2a
_sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
_sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2a
_sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2e
csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2e
csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2e
csequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
dsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0lsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2f
dsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
^sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Ysequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumksequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2[
Ysequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
bsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2d
bsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
^sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Ysequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ksequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_sequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2[
Ysequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
Ksequential/vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Xsequential_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle7sequential/vectorize/StringSplit/StringSplitV2:values:0Ysequential_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2M
Ksequential/vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2?
4sequential/vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 26
4sequential/vectorize/string_lookup/assert_equal/NoOp?
+sequential/vectorize/string_lookup/IdentityIdentityTsequential/vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2-
+sequential/vectorize/string_lookup/Identity?
-sequential/vectorize/string_lookup/Identity_1Identitybsequential/vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????2/
-sequential/vectorize/string_lookup/Identity_1?
1sequential/vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 23
1sequential/vectorize/RaggedToTensor/default_value?
)sequential/vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2+
)sequential/vectorize/RaggedToTensor/Const?
8sequential/vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2sequential/vectorize/RaggedToTensor/Const:output:04sequential/vectorize/string_lookup/Identity:output:0:sequential/vectorize/RaggedToTensor/default_value:output:06sequential/vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2:
8sequential/vectorize/RaggedToTensor/RaggedTensorToTensor?
sequential/vectorize/ShapeShapeAsequential/vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
sequential/vectorize/Shape?
(sequential/vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/vectorize/strided_slice/stack?
*sequential/vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential/vectorize/strided_slice/stack_1?
*sequential/vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential/vectorize/strided_slice/stack_2?
"sequential/vectorize/strided_sliceStridedSlice#sequential/vectorize/Shape:output:01sequential/vectorize/strided_slice/stack:output:03sequential/vectorize/strided_slice/stack_1:output:03sequential/vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential/vectorize/strided_slice{
sequential/vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
sequential/vectorize/sub/x?
sequential/vectorize/subSub#sequential/vectorize/sub/x:output:0+sequential/vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
sequential/vectorize/sub}
sequential/vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
sequential/vectorize/Less/y?
sequential/vectorize/LessLess+sequential/vectorize/strided_slice:output:0$sequential/vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/vectorize/Less?
sequential/vectorize/condStatelessIfsequential/vectorize/Less:z:0sequential/vectorize/sub:z:0Asequential/vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *8
else_branch)R'
%sequential_vectorize_cond_false_16975*/
output_shapes
:??????????????????*7
then_branch(R&
$sequential_vectorize_cond_true_169742
sequential/vectorize/cond?
"sequential/vectorize/cond/IdentityIdentity"sequential/vectorize/cond:output:0*
T0	*(
_output_shapes
:??????????2$
"sequential/vectorize/cond/Identity?
4sequential/embedding/embedding_lookup/ReadVariableOpReadVariableOp=sequential_embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
??@*
dtype026
4sequential/embedding/embedding_lookup/ReadVariableOp?
*sequential/embedding/embedding_lookup/axisConst*G
_class=
;9loc:@sequential/embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/embedding/embedding_lookup/axis?
%sequential/embedding/embedding_lookupGatherV2<sequential/embedding/embedding_lookup/ReadVariableOp:value:0+sequential/vectorize/cond/Identity:output:03sequential/embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*G
_class=
;9loc:@sequential/embedding/embedding_lookup/ReadVariableOp*,
_output_shapes
:??????????@2'
%sequential/embedding/embedding_lookup?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*,
_output_shapes
:??????????@20
.sequential/embedding/embedding_lookup/Identity?
:sequential/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential/global_average_pooling1d/Mean/reduction_indices?
(sequential/global_average_pooling1d/MeanMean7sequential/embedding/embedding_lookup/Identity:output:0Csequential/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2*
(sequential/global_average_pooling1d/Mean?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul1sequential/global_average_pooling1d/Mean:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential/dense/Relu?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/Softmax?
IdentityIdentity$sequential/dense_1/Softmax:softmax:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp5^sequential/embedding/embedding_lookup/ReadVariableOpL^sequential/vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2l
4sequential/embedding/embedding_lookup/ReadVariableOp4sequential/embedding/embedding_lookup/ReadVariableOp2?
Ksequential/vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2Ksequential/vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:X T
'
_output_shapes
:?????????
)
_user_specified_namevectorize_input:

_output_shapes
: 
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_17292
vectorize_inputQ
Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleR
Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
embedding_17277
dense_17281
dense_17283
dense_1_17286
dense_1_17288
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2?
vectorize/SqueezeSqueezevectorize_input*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
vectorize/Squeeze{
vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
vectorize/StringSplit/Const?
#vectorize/StringSplit/StringSplitV2StringSplitV2vectorize/Squeeze:output:0$vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2%
#vectorize/StringSplit/StringSplitV2?
)vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)vectorize/StringSplit/strided_slice/stack?
+vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+vectorize/StringSplit/strided_slice/stack_1?
+vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+vectorize/StringSplit/strided_slice/stack_2?
#vectorize/StringSplit/strided_sliceStridedSlice-vectorize/StringSplit/StringSplitV2:indices:02vectorize/StringSplit/strided_slice/stack:output:04vectorize/StringSplit/strided_slice/stack_1:output:04vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2%
#vectorize/StringSplit/strided_slice?
+vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+vectorize/StringSplit/strided_slice_1/stack?
-vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_1?
-vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_2?
%vectorize/StringSplit/strided_slice_1StridedSlice+vectorize/StringSplit/StringSplitV2:shape:04vectorize/StringSplit/strided_slice_1/stack:output:06vectorize/StringSplit/strided_slice_1/stack_1:output:06vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2'
%vectorize/StringSplit/strided_slice_1?
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast,vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2N
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast.vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapePvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProd_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreater^vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0cvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2]vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulYvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2[
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsum`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2Y
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle,vectorize/StringSplit/StringSplitV2:values:0Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2t
)vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2+
)vectorize/string_lookup/assert_equal/NoOp?
 vectorize/string_lookup/IdentityIdentityIvectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2"
 vectorize/string_lookup/Identity?
"vectorize/string_lookup/Identity_1IdentityWvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????2$
"vectorize/string_lookup/Identity_1?
&vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2(
&vectorize/RaggedToTensor/default_value?
vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2 
vectorize/RaggedToTensor/Const?
-vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor'vectorize/RaggedToTensor/Const:output:0)vectorize/string_lookup/Identity:output:0/vectorize/RaggedToTensor/default_value:output:0+vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2/
-vectorize/RaggedToTensor/RaggedTensorToTensor?
vectorize/ShapeShape6vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
vectorize/Shape?
vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
vectorize/strided_slice/stack?
vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_1?
vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_2?
vectorize/strided_sliceStridedSlicevectorize/Shape:output:0&vectorize/strided_slice/stack:output:0(vectorize/strided_slice/stack_1:output:0(vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vectorize/strided_slicee
vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/sub/x?
vectorize/subSubvectorize/sub/x:output:0 vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
vectorize/subg
vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/Less/y?
vectorize/LessLess vectorize/strided_slice:output:0vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
vectorize/Less?
vectorize/condStatelessIfvectorize/Less:z:0vectorize/sub:z:06vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
else_branchR
vectorize_cond_false_17257*/
output_shapes
:??????????????????*,
then_branchR
vectorize_cond_true_172562
vectorize/cond?
vectorize/cond/IdentityIdentityvectorize/cond:output:0*
T0	*(
_output_shapes
:??????????2
vectorize/cond/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCall vectorize/cond/Identity:output:0embedding_17277*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_171222#
!embedding/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_171392*
(global_average_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_17281dense_17283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_171572
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_17286dense_1_17288*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_171842!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCallA^vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:X T
'
_output_shapes
:?????????
)
_user_specified_namevectorize_input:

_output_shapes
: 
?
o
)__inference_embedding_layer_call_fn_17819

inputs	
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_171222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
vectorize_cond_false_17461
vectorize_cond_placeholderN
Jvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
"vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"vectorize/cond/strided_slice/stack?
$vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2&
$vectorize/cond/strided_slice/stack_1?
$vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$vectorize/cond/strided_slice/stack_2?
vectorize/cond/strided_sliceStridedSliceJvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor+vectorize/cond/strided_slice/stack:output:0-vectorize/cond/strided_slice/stack_1:output:0-vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2
vectorize/cond/strided_slice?
vectorize/cond/IdentityIdentity%vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
.
__inference__initializer_17891
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
vectorize_cond_true_17460/
+vectorize_cond_pad_paddings_1_vectorize_subD
@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2!
vectorize/cond/Pad/paddings/1/0?
vectorize/cond/Pad/paddings/1Pack(vectorize/cond/Pad/paddings/1/0:output:0+vectorize_cond_pad_paddings_1_vectorize_sub*
N*
T0*
_output_shapes
:2
vectorize/cond/Pad/paddings/1?
vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
vectorize/cond/Pad/paddings/0_1?
vectorize/cond/Pad/paddingsPack(vectorize/cond/Pad/paddings/0_1:output:0&vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2
vectorize/cond/Pad/paddings?
vectorize/cond/PadPad@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor$vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Pad?
vectorize/cond/IdentityIdentityvectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
#__inference_signature_wrapper_17542
vectorize_input
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallvectorize_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_170172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namevectorize_input:

_output_shapes
: 
?
?
vectorize_cond_false_17257
vectorize_cond_placeholderN
Jvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
"vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"vectorize/cond/strided_slice/stack?
$vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2&
$vectorize/cond/strided_slice/stack_1?
$vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$vectorize/cond/strided_slice/stack_2?
vectorize/cond/strided_sliceStridedSliceJvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor+vectorize/cond/strided_slice/stack:output:0-vectorize/cond/strided_slice/stack_1:output:0-vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2
vectorize/cond/strided_slice?
vectorize/cond/IdentityIdentity%vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
%sequential_vectorize_cond_false_16975)
%sequential_vectorize_cond_placeholderd
`sequential_vectorize_cond_strided_slice_sequential_vectorize_raggedtotensor_raggedtensortotensor	&
"sequential_vectorize_cond_identity	?
-sequential/vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-sequential/vectorize/cond/strided_slice/stack?
/sequential/vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  21
/sequential/vectorize/cond/strided_slice/stack_1?
/sequential/vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/vectorize/cond/strided_slice/stack_2?
'sequential/vectorize/cond/strided_sliceStridedSlice`sequential_vectorize_cond_strided_slice_sequential_vectorize_raggedtotensor_raggedtensortotensor6sequential/vectorize/cond/strided_slice/stack:output:08sequential/vectorize/cond/strided_slice/stack_1:output:08sequential/vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2)
'sequential/vectorize/cond/strided_slice?
"sequential/vectorize/cond/IdentityIdentity0sequential/vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2$
"sequential/vectorize/cond/Identity"Q
"sequential_vectorize_cond_identity+sequential/vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
z
%__inference_dense_layer_call_fn_17861

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_171572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
,
__inference__destroyer_17896
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
vectorize_cond_true_17350/
+vectorize_cond_pad_paddings_1_vectorize_subD
@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2!
vectorize/cond/Pad/paddings/1/0?
vectorize/cond/Pad/paddings/1Pack(vectorize/cond/Pad/paddings/1/0:output:0+vectorize_cond_pad_paddings_1_vectorize_sub*
N*
T0*
_output_shapes
:2
vectorize/cond/Pad/paddings/1?
vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
vectorize/cond/Pad/paddings/0_1?
vectorize/cond/Pad/paddingsPack(vectorize/cond/Pad/paddings/0_1:output:0&vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2
vectorize/cond/Pad/paddings?
vectorize/cond/PadPad@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor$vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Pad?
vectorize/cond/IdentityIdentityvectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
vectorize_cond_true_17722/
+vectorize_cond_pad_paddings_1_vectorize_subD
@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2!
vectorize/cond/Pad/paddings/1/0?
vectorize/cond/Pad/paddings/1Pack(vectorize/cond/Pad/paddings/1/0:output:0+vectorize_cond_pad_paddings_1_vectorize_sub*
N*
T0*
_output_shapes
:2
vectorize/cond/Pad/paddings/1?
vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
vectorize/cond/Pad/paddings/0_1?
vectorize/cond/Pad/paddingsPack(vectorize/cond/Pad/paddings/0_1:output:0&vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2
vectorize/cond/Pad/paddings?
vectorize/cond/PadPad@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor$vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Pad?
vectorize/cond/IdentityIdentityvectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
vectorize_cond_false_17625
vectorize_cond_placeholderN
Jvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
"vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"vectorize/cond/strided_slice/stack?
$vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2&
$vectorize/cond/strided_slice/stack_1?
$vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$vectorize/cond/strided_slice/stack_2?
vectorize/cond/strided_sliceStridedSliceJvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor+vectorize/cond/strided_slice/stack:output:0-vectorize/cond/strided_slice/stack_1:output:0-vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2
vectorize/cond/strided_slice?
vectorize/cond/IdentityIdentity%vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_17201
vectorize_inputQ
Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleR
Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
embedding_17131
dense_17168
dense_17170
dense_1_17195
dense_1_17197
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2?
vectorize/SqueezeSqueezevectorize_input*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
vectorize/Squeeze{
vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
vectorize/StringSplit/Const?
#vectorize/StringSplit/StringSplitV2StringSplitV2vectorize/Squeeze:output:0$vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2%
#vectorize/StringSplit/StringSplitV2?
)vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)vectorize/StringSplit/strided_slice/stack?
+vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+vectorize/StringSplit/strided_slice/stack_1?
+vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+vectorize/StringSplit/strided_slice/stack_2?
#vectorize/StringSplit/strided_sliceStridedSlice-vectorize/StringSplit/StringSplitV2:indices:02vectorize/StringSplit/strided_slice/stack:output:04vectorize/StringSplit/strided_slice/stack_1:output:04vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2%
#vectorize/StringSplit/strided_slice?
+vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+vectorize/StringSplit/strided_slice_1/stack?
-vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_1?
-vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_2?
%vectorize/StringSplit/strided_slice_1StridedSlice+vectorize/StringSplit/StringSplitV2:shape:04vectorize/StringSplit/strided_slice_1/stack:output:06vectorize/StringSplit/strided_slice_1/stack_1:output:06vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2'
%vectorize/StringSplit/strided_slice_1?
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast,vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2N
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast.vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapePvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProd_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreater^vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0cvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2]vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulYvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2[
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsum`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2Y
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle,vectorize/StringSplit/StringSplitV2:values:0Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2t
)vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2+
)vectorize/string_lookup/assert_equal/NoOp?
 vectorize/string_lookup/IdentityIdentityIvectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2"
 vectorize/string_lookup/Identity?
"vectorize/string_lookup/Identity_1IdentityWvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????2$
"vectorize/string_lookup/Identity_1?
&vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2(
&vectorize/RaggedToTensor/default_value?
vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2 
vectorize/RaggedToTensor/Const?
-vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor'vectorize/RaggedToTensor/Const:output:0)vectorize/string_lookup/Identity:output:0/vectorize/RaggedToTensor/default_value:output:0+vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2/
-vectorize/RaggedToTensor/RaggedTensorToTensor?
vectorize/ShapeShape6vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
vectorize/Shape?
vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
vectorize/strided_slice/stack?
vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_1?
vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_2?
vectorize/strided_sliceStridedSlicevectorize/Shape:output:0&vectorize/strided_slice/stack:output:0(vectorize/strided_slice/stack_1:output:0(vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vectorize/strided_slicee
vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/sub/x?
vectorize/subSubvectorize/sub/x:output:0 vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
vectorize/subg
vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/Less/y?
vectorize/LessLess vectorize/strided_slice:output:0vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
vectorize/Less?
vectorize/condStatelessIfvectorize/Less:z:0vectorize/sub:z:06vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
else_branchR
vectorize_cond_false_17093*/
output_shapes
:??????????????????*,
then_branchR
vectorize_cond_true_170922
vectorize/cond?
vectorize/cond/IdentityIdentityvectorize/cond:output:0*
T0	*(
_output_shapes
:??????????2
vectorize/cond/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCall vectorize/cond/Identity:output:0embedding_17131*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_171222#
!embedding/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_171392*
(global_average_pooling1d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_17168dense_17170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_171572
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_17195dense_1_17197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_171842!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCallA^vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:X T
'
_output_shapes
:?????????
)
_user_specified_namevectorize_input:

_output_shapes
: 
?

?
D__inference_embedding_layer_call_and_return_conditional_losses_17122

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
??@*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*,
_output_shapes
:??????????@2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_output_shapes
:??????????@2
embedding_lookup/Identity?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_17836

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
$sequential_vectorize_cond_true_16974E
Asequential_vectorize_cond_pad_paddings_1_sequential_vectorize_subZ
Vsequential_vectorize_cond_pad_sequential_vectorize_raggedtotensor_raggedtensortotensor	&
"sequential_vectorize_cond_identity	?
*sequential/vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/vectorize/cond/Pad/paddings/1/0?
(sequential/vectorize/cond/Pad/paddings/1Pack3sequential/vectorize/cond/Pad/paddings/1/0:output:0Asequential_vectorize_cond_pad_paddings_1_sequential_vectorize_sub*
N*
T0*
_output_shapes
:2*
(sequential/vectorize/cond/Pad/paddings/1?
*sequential/vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*sequential/vectorize/cond/Pad/paddings/0_1?
&sequential/vectorize/cond/Pad/paddingsPack3sequential/vectorize/cond/Pad/paddings/0_1:output:01sequential/vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2(
&sequential/vectorize/cond/Pad/paddings?
sequential/vectorize/cond/PadPadVsequential_vectorize_cond_pad_sequential_vectorize_raggedtotensor_raggedtensortotensor/sequential/vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
sequential/vectorize/cond/Pad?
"sequential/vectorize/cond/IdentityIdentity&sequential/vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2$
"sequential/vectorize/cond/Identity"Q
"sequential_vectorize_cond_identity+sequential/vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
vectorize_cond_false_17723
vectorize_cond_placeholderN
Jvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
"vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"vectorize/cond/strided_slice/stack?
$vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  2&
$vectorize/cond/strided_slice/stack_1?
$vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$vectorize/cond/strided_slice/stack_2?
vectorize/cond/strided_sliceStridedSliceJvectorize_cond_strided_slice_vectorize_raggedtotensor_raggedtensortotensor+vectorize/cond/strided_slice/stack:output:0-vectorize/cond/strided_slice/stack_1:output:0-vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:??????????????????*

begin_mask*
end_mask2
vectorize/cond/strided_slice?
vectorize/cond/IdentityIdentity%vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?
?
*__inference_sequential_layer_call_fn_17403
vectorize_input
unknown
	unknown_0	
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallvectorize_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_173862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????
)
_user_specified_namevectorize_input:

_output_shapes
: 
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_17667

inputsQ
Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleR
Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	6
2embedding_embedding_lookup_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?)embedding/embedding_lookup/ReadVariableOp?@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2?
vectorize/SqueezeSqueezeinputs*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????2
vectorize/Squeeze{
vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
vectorize/StringSplit/Const?
#vectorize/StringSplit/StringSplitV2StringSplitV2vectorize/Squeeze:output:0$vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2%
#vectorize/StringSplit/StringSplitV2?
)vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)vectorize/StringSplit/strided_slice/stack?
+vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+vectorize/StringSplit/strided_slice/stack_1?
+vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+vectorize/StringSplit/strided_slice/stack_2?
#vectorize/StringSplit/strided_sliceStridedSlice-vectorize/StringSplit/StringSplitV2:indices:02vectorize/StringSplit/strided_slice/stack:output:04vectorize/StringSplit/strided_slice/stack_1:output:04vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2%
#vectorize/StringSplit/strided_slice?
+vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+vectorize/StringSplit/strided_slice_1/stack?
-vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_1?
-vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-vectorize/StringSplit/strided_slice_1/stack_2?
%vectorize/StringSplit/strided_slice_1StridedSlice+vectorize/StringSplit/StringSplitV2:shape:04vectorize/StringSplit/strided_slice_1/stack:output:06vectorize/StringSplit/strided_slice_1/stack_1:output:06vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2'
%vectorize/StringSplit/strided_slice_1?
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast,vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2N
Lvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast.vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapePvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProd_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreater^vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0cvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2W
Uvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2X
Vvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2]vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulYvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2V
Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumRvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2Z
Xvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountPvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0avectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2[
Yvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsum`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2Y
Wvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Svectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2`vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Tvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0\vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2P
Nvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Mvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle,vectorize/StringSplit/StringSplitV2:values:0Nvectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:?????????2B
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2t
)vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2+
)vectorize/string_lookup/assert_equal/NoOp?
 vectorize/string_lookup/IdentityIdentityIvectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2"
 vectorize/string_lookup/Identity?
"vectorize/string_lookup/Identity_1IdentityWvectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:?????????2$
"vectorize/string_lookup/Identity_1?
&vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2(
&vectorize/RaggedToTensor/default_value?
vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2 
vectorize/RaggedToTensor/Const?
-vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor'vectorize/RaggedToTensor/Const:output:0)vectorize/string_lookup/Identity:output:0/vectorize/RaggedToTensor/default_value:output:0+vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS2/
-vectorize/RaggedToTensor/RaggedTensorToTensor?
vectorize/ShapeShape6vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
vectorize/Shape?
vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
vectorize/strided_slice/stack?
vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_1?
vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
vectorize/strided_slice/stack_2?
vectorize/strided_sliceStridedSlicevectorize/Shape:output:0&vectorize/strided_slice/stack:output:0(vectorize/strided_slice/stack_1:output:0(vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vectorize/strided_slicee
vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/sub/x?
vectorize/subSubvectorize/sub/x:output:0 vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
vectorize/subg
vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
vectorize/Less/y?
vectorize/LessLess vectorize/strided_slice:output:0vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
vectorize/Less?
vectorize/condStatelessIfvectorize/Less:z:0vectorize/sub:z:06vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
else_branchR
vectorize_cond_false_17625*/
output_shapes
:??????????????????*,
then_branchR
vectorize_cond_true_176242
vectorize/cond?
vectorize/cond/IdentityIdentityvectorize/cond:output:0*
T0	*(
_output_shapes
:??????????2
vectorize/cond/Identity?
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
??@*
dtype02+
)embedding/embedding_lookup/ReadVariableOp?
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axis?
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0 vectorize/cond/Identity:output:0(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*,
_output_shapes
:??????????@2
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*,
_output_shapes
:??????????@2%
#embedding/embedding_lookup/Identity?
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices?
global_average_pooling1d/MeanMean,embedding/embedding_lookup/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
global_average_pooling1d/Mean?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
IdentityIdentitydense_1/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^embedding/embedding_lookup/ReadVariableOpA^vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:?????????:: :::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)embedding/embedding_lookup/ReadVariableOp)embedding/embedding_lookup/ReadVariableOp2?
@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2@vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
vectorize_cond_true_17624/
+vectorize_cond_pad_paddings_1_vectorize_subD
@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor	
vectorize_cond_identity	?
vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2!
vectorize/cond/Pad/paddings/1/0?
vectorize/cond/Pad/paddings/1Pack(vectorize/cond/Pad/paddings/1/0:output:0+vectorize_cond_pad_paddings_1_vectorize_sub*
N*
T0*
_output_shapes
:2
vectorize/cond/Pad/paddings/1?
vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
vectorize/cond/Pad/paddings/0_1?
vectorize/cond/Pad/paddingsPack(vectorize/cond/Pad/paddings/0_1:output:0&vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2
vectorize/cond/Pad/paddings?
vectorize/cond/PadPad@vectorize_cond_pad_vectorize_raggedtotensor_raggedtensortotensor$vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Pad?
vectorize/cond/IdentityIdentityvectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:??????????????????2
vectorize/cond/Identity";
vectorize_cond_identity vectorize/cond/Identity:output:0*1
_input_shapes 
: :??????????????????: 

_output_shapes
: :62
0
_output_shapes
:??????????????????
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_17852

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
vectorize_input8
!serving_default_vectorize_input:0?????????;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?(
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
Z__call__
[_default_save_signature
*\&call_and_return_all_conditional_losses"?%
_tf_keras_sequential?%{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "vectorize_input"}}, {"class_name": "TextVectorization", "config": {"name": "vectorize", "trainable": true, "dtype": "string", "max_tokens": 40000, "standardize": null, "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 300, "pad_to_max_tokens": true}}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 40000, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "vectorize_input"}}, {"class_name": "TextVectorization", "config": {"name": "vectorize", "trainable": true, "dtype": "string", "max_tokens": 40000, "standardize": null, "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 300, "pad_to_max_tokens": true}}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 40000, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
state_variables
_index_lookup_layer
	keras_api"?
_tf_keras_layer?{"class_name": "TextVectorization", "name": "vectorize", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "vectorize", "trainable": true, "dtype": "string", "max_tokens": 40000, "standardize": null, "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 300, "pad_to_max_tokens": true}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 40000, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
e__call__
*f&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
$iter

%beta_1

&beta_2
	'decay
(learning_ratemPmQmRmSmTvUvVvWvXvY"
	optimizer
 "
trackable_list_wrapper
C
1
2
3
4
5"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
regularization_losses

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
-non_trainable_variables
	trainable_variables
Z__call__
[_default_save_signature
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
,
gserving_default"
signature_map
 "
trackable_dict_wrapper
?
.state_variables

/_table
0	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "string", "invert": false, "max_tokens": 40000, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
"
_generic_user_object
(:&
??@2embedding/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
regularization_losses

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
5non_trainable_variables
trainable_variables
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
:non_trainable_variables
trainable_variables
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
:@@2dense/kernel
:@2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
?non_trainable_variables
trainable_variables
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
 regularization_losses

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
!	variables
Dnon_trainable_variables
"trainable_variables
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
C
0
1
2
3
4"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
O
h_create_resource
i_initialize
j_destroy_resourceR Z
table]^
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	Gtotal
	Hcount
I	variables
J	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	Ktotal
	Lcount
M
_fn_kwargs
N	variables
O	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
G0
H1"
trackable_list_wrapper
-
I	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
K0
L1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
-:+
??@2Adam/embedding/embeddings/m
#:!@@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
%:#@2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
-:+
??@2Adam/embedding/embeddings/v
#:!@@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
%:#@2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
?2?
*__inference_sequential_layer_call_fn_17803
*__inference_sequential_layer_call_fn_17784
*__inference_sequential_layer_call_fn_17513
*__inference_sequential_layer_call_fn_17403?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_17017?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
vectorize_input?????????
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_17292
E__inference_sequential_layer_call_and_return_conditional_losses_17765
E__inference_sequential_layer_call_and_return_conditional_losses_17667
E__inference_sequential_layer_call_and_return_conditional_losses_17201?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference_save_fn_17915checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_17923restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
?2?
)__inference_embedding_layer_call_fn_17819?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_embedding_layer_call_and_return_conditional_losses_17812?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_global_average_pooling1d_layer_call_fn_17830
8__inference_global_average_pooling1d_layer_call_fn_17841?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_17836
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_17825?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_17861?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_17852?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_17881?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_17872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_17542vectorize_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_17886?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_17891?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_17896?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
	J
Const6
__inference__creator_17886?

? 
? "? 8
__inference__destroyer_17896?

? 
? "? :
__inference__initializer_17891?

? 
? "? ?
 __inference__wrapped_model_17017v/k8?5
.?+
)?&
vectorize_input?????????
? "1?.
,
dense_1!?
dense_1??????????
B__inference_dense_1_layer_call_and_return_conditional_losses_17872\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_17881O/?,
%?"
 ?
inputs?????????@
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_17852\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? x
%__inference_dense_layer_call_fn_17861O/?,
%?"
 ?
inputs?????????@
? "??????????@?
D__inference_embedding_layer_call_and_return_conditional_losses_17812a0?-
&?#
!?
inputs??????????	
? "*?'
 ?
0??????????@
? ?
)__inference_embedding_layer_call_fn_17819T0?-
&?#
!?
inputs??????????	
? "???????????@?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_17825a8?5
.?+
%?"
inputs??????????@

 
? "%?"
?
0?????????@
? ?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_17836{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
8__inference_global_average_pooling1d_layer_call_fn_17830T8?5
.?+
%?"
inputs??????????@

 
? "??????????@?
8__inference_global_average_pooling1d_layer_call_fn_17841nI?F
??<
6?3
inputs'???????????????????????????

 
? "!???????????????????y
__inference_restore_fn_17923Y/K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_17915?/&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
E__inference_sequential_layer_call_and_return_conditional_losses_17201r/k@?=
6?3
)?&
vectorize_input?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_17292r/k@?=
6?3
)?&
vectorize_input?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_17667i/k7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_17765i/k7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_17403e/k@?=
6?3
)?&
vectorize_input?????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_17513e/k@?=
6?3
)?&
vectorize_input?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_17784\/k7?4
-?*
 ?
inputs?????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_17803\/k7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
#__inference_signature_wrapper_17542?/kK?H
? 
A?>
<
vectorize_input)?&
vectorize_input?????????"1?.
,
dense_1!?
dense_1?????????