??'
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

.
Identity

input"T
output"T"	
Ttype
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
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
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
executor_typestring ??
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.22v2.8.2-0-g2ea19cbb5758??"
?
gsc_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namegsc_2/kernel
y
 gsc_2/kernel/Read/ReadVariableOpReadVariableOpgsc_2/kernel**
_output_shapes
:*
dtype0
l

gsc_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
gsc_2/bias
e
gsc_2/bias/Read/ReadVariableOpReadVariableOp
gsc_2/bias*
_output_shapes
:*
dtype0
?
opt_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameopt_2/kernel
y
 opt_2/kernel/Read/ReadVariableOpReadVariableOpopt_2/kernel**
_output_shapes
:*
dtype0
l

opt_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
opt_2/bias
e
opt_2/bias/Read/ReadVariableOpReadVariableOp
opt_2/bias*
_output_shapes
:*
dtype0
?
gsc_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namegsc_3/kernel
y
 gsc_3/kernel/Read/ReadVariableOpReadVariableOpgsc_3/kernel**
_output_shapes
:*
dtype0
l

gsc_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
gsc_3/bias
e
gsc_3/bias/Read/ReadVariableOpReadVariableOp
gsc_3/bias*
_output_shapes
:*
dtype0
?
opt_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameopt_3/kernel
y
 opt_3/kernel/Read/ReadVariableOpReadVariableOpopt_3/kernel**
_output_shapes
:*
dtype0
l

opt_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
opt_3/bias
e
opt_3/bias/Read/ReadVariableOpReadVariableOp
opt_3/bias*
_output_shapes
:*
dtype0
?
gsc_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namegsc_5/kernel
y
 gsc_5/kernel/Read/ReadVariableOpReadVariableOpgsc_5/kernel**
_output_shapes
:*
dtype0
l

gsc_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
gsc_5/bias
e
gsc_5/bias/Read/ReadVariableOpReadVariableOp
gsc_5/bias*
_output_shapes
:*
dtype0
?
opt_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameopt_5/kernel
y
 opt_5/kernel/Read/ReadVariableOpReadVariableOpopt_5/kernel**
_output_shapes
:*
dtype0
l

opt_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
opt_5/bias
e
opt_5/bias/Read/ReadVariableOpReadVariableOp
opt_5/bias*
_output_shapes
:*
dtype0
?
gsc_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namegsc_6/kernel
y
 gsc_6/kernel/Read/ReadVariableOpReadVariableOpgsc_6/kernel**
_output_shapes
:*
dtype0
l

gsc_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
gsc_6/bias
e
gsc_6/bias/Read/ReadVariableOpReadVariableOp
gsc_6/bias*
_output_shapes
:*
dtype0
?
opt_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameopt_6/kernel
y
 opt_6/kernel/Read/ReadVariableOpReadVariableOpopt_6/kernel**
_output_shapes
:*
dtype0
l

opt_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
opt_6/bias
e
opt_6/bias/Read/ReadVariableOpReadVariableOp
opt_6/bias*
_output_shapes
:*
dtype0
?
gsc_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namegsc_8/kernel
y
 gsc_8/kernel/Read/ReadVariableOpReadVariableOpgsc_8/kernel**
_output_shapes
: *
dtype0
l

gsc_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
gsc_8/bias
e
gsc_8/bias/Read/ReadVariableOpReadVariableOp
gsc_8/bias*
_output_shapes
: *
dtype0
?
opt_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameopt_8/kernel
y
 opt_8/kernel/Read/ReadVariableOpReadVariableOpopt_8/kernel**
_output_shapes
: *
dtype0
l

opt_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
opt_8/bias
e
opt_8/bias/Read/ReadVariableOpReadVariableOp
opt_8/bias*
_output_shapes
: *
dtype0
?
gsc_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_namegsc_9/kernel
y
 gsc_9/kernel/Read/ReadVariableOpReadVariableOpgsc_9/kernel**
_output_shapes
:  *
dtype0
l

gsc_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
gsc_9/bias
e
gsc_9/bias/Read/ReadVariableOpReadVariableOp
gsc_9/bias*
_output_shapes
: *
dtype0
?
opt_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameopt_9/kernel
y
 opt_9/kernel/Read/ReadVariableOpReadVariableOpopt_9/kernel**
_output_shapes
:  *
dtype0
l

opt_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
opt_9/bias
e
opt_9/bias/Read/ReadVariableOpReadVariableOp
opt_9/bias*
_output_shapes
: *
dtype0
?
gsc_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_namegsc_11/kernel
{
!gsc_11/kernel/Read/ReadVariableOpReadVariableOpgsc_11/kernel**
_output_shapes
:  *
dtype0
n
gsc_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namegsc_11/bias
g
gsc_11/bias/Read/ReadVariableOpReadVariableOpgsc_11/bias*
_output_shapes
: *
dtype0
?
opt_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameopt_11/kernel
{
!opt_11/kernel/Read/ReadVariableOpReadVariableOpopt_11/kernel**
_output_shapes
:  *
dtype0
n
opt_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameopt_11/bias
g
opt_11/bias/Read/ReadVariableOpReadVariableOpopt_11/bias*
_output_shapes
: *
dtype0
?
gsc_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_namegsc_12/kernel
{
!gsc_12/kernel/Read/ReadVariableOpReadVariableOpgsc_12/kernel**
_output_shapes
:  *
dtype0
n
gsc_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namegsc_12/bias
g
gsc_12/bias/Read/ReadVariableOpReadVariableOpgsc_12/bias*
_output_shapes
: *
dtype0
?
opt_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameopt_12/kernel
{
!opt_12/kernel/Read/ReadVariableOpReadVariableOpopt_12/kernel**
_output_shapes
:  *
dtype0
n
opt_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameopt_12/bias
g
opt_12/bias/Read/ReadVariableOpReadVariableOpopt_12/bias*
_output_shapes
: *
dtype0
?
merge_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_namemerge_1/kernel
}
"merge_1/kernel/Read/ReadVariableOpReadVariableOpmerge_1/kernel**
_output_shapes
: @*
dtype0
p
merge_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namemerge_1/bias
i
 merge_1/bias/Read/ReadVariableOpReadVariableOpmerge_1/bias*
_output_shapes
:@*
dtype0
?
merge_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_namemerge_2/kernel
}
"merge_2/kernel/Read/ReadVariableOpReadVariableOpmerge_2/kernel**
_output_shapes
:@@*
dtype0
p
merge_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namemerge_2/bias
i
 merge_2/bias/Read/ReadVariableOpReadVariableOpmerge_2/bias*
_output_shapes
:@*
dtype0
?
merge_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_namemerge_4/kernel
}
"merge_4/kernel/Read/ReadVariableOpReadVariableOpmerge_4/kernel**
_output_shapes
:@@*
dtype0
p
merge_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namemerge_4/bias
i
 merge_4/bias/Read/ReadVariableOpReadVariableOpmerge_4/bias*
_output_shapes
:@*
dtype0
?
merge_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_namemerge_5/kernel
}
"merge_5/kernel/Read/ReadVariableOpReadVariableOpmerge_5/kernel**
_output_shapes
:@@*
dtype0
p
merge_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namemerge_5/bias
i
 merge_5/bias/Read/ReadVariableOpReadVariableOpmerge_5/bias*
_output_shapes
:@*
dtype0
?
merge_7/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?*
shared_namemerge_7/kernel
~
"merge_7/kernel/Read/ReadVariableOpReadVariableOpmerge_7/kernel*+
_output_shapes
:@?*
dtype0
q
merge_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namemerge_7/bias
j
 merge_7/bias/Read/ReadVariableOpReadVariableOpmerge_7/bias*
_output_shapes	
:?*
dtype0
?
merge_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??*
shared_namemerge_8/kernel

"merge_8/kernel/Read/ReadVariableOpReadVariableOpmerge_8/kernel*,
_output_shapes
:??*
dtype0
q
merge_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namemerge_8/bias
j
 merge_8/bias/Read/ReadVariableOpReadVariableOpmerge_8/bias*
_output_shapes	
:?*
dtype0
t
fc_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namefc_1/kernel
m
fc_1/kernel/Read/ReadVariableOpReadVariableOpfc_1/kernel* 
_output_shapes
:
??*
dtype0
k
	fc_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	fc_1/bias
d
fc_1/bias/Read/ReadVariableOpReadVariableOp	fc_1/bias*
_output_shapes	
:?*
dtype0
s
fc_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *
shared_namefc_3/kernel
l
fc_3/kernel/Read/ReadVariableOpReadVariableOpfc_3/kernel*
_output_shapes
:	? *
dtype0
j
	fc_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	fc_3/bias
c
fc_3/bias/Read/ReadVariableOpReadVariableOp	fc_3/bias*
_output_shapes
: *
dtype0
r
pred/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namepred/kernel
k
pred/kernel/Read/ReadVariableOpReadVariableOppred/kernel*
_output_shapes

: *
dtype0
j
	pred/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	pred/bias
c
pred/bias/Read/ReadVariableOpReadVariableOp	pred/bias*
_output_shapes
:*
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
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
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
Adam/gsc_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/gsc_2/kernel/m
?
'Adam/gsc_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gsc_2/kernel/m**
_output_shapes
:*
dtype0
z
Adam/gsc_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/gsc_2/bias/m
s
%Adam/gsc_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/gsc_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/opt_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/opt_2/kernel/m
?
'Adam/opt_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/opt_2/kernel/m**
_output_shapes
:*
dtype0
z
Adam/opt_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/opt_2/bias/m
s
%Adam/opt_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/opt_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/gsc_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/gsc_3/kernel/m
?
'Adam/gsc_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gsc_3/kernel/m**
_output_shapes
:*
dtype0
z
Adam/gsc_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/gsc_3/bias/m
s
%Adam/gsc_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/gsc_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/opt_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/opt_3/kernel/m
?
'Adam/opt_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/opt_3/kernel/m**
_output_shapes
:*
dtype0
z
Adam/opt_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/opt_3/bias/m
s
%Adam/opt_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/opt_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/gsc_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/gsc_5/kernel/m
?
'Adam/gsc_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gsc_5/kernel/m**
_output_shapes
:*
dtype0
z
Adam/gsc_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/gsc_5/bias/m
s
%Adam/gsc_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/gsc_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/opt_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/opt_5/kernel/m
?
'Adam/opt_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/opt_5/kernel/m**
_output_shapes
:*
dtype0
z
Adam/opt_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/opt_5/bias/m
s
%Adam/opt_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/opt_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/gsc_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/gsc_6/kernel/m
?
'Adam/gsc_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gsc_6/kernel/m**
_output_shapes
:*
dtype0
z
Adam/gsc_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/gsc_6/bias/m
s
%Adam/gsc_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/gsc_6/bias/m*
_output_shapes
:*
dtype0
?
Adam/opt_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/opt_6/kernel/m
?
'Adam/opt_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/opt_6/kernel/m**
_output_shapes
:*
dtype0
z
Adam/opt_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/opt_6/bias/m
s
%Adam/opt_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/opt_6/bias/m*
_output_shapes
:*
dtype0
?
Adam/gsc_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/gsc_8/kernel/m
?
'Adam/gsc_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gsc_8/kernel/m**
_output_shapes
: *
dtype0
z
Adam/gsc_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/gsc_8/bias/m
s
%Adam/gsc_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/gsc_8/bias/m*
_output_shapes
: *
dtype0
?
Adam/opt_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/opt_8/kernel/m
?
'Adam/opt_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/opt_8/kernel/m**
_output_shapes
: *
dtype0
z
Adam/opt_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/opt_8/bias/m
s
%Adam/opt_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/opt_8/bias/m*
_output_shapes
: *
dtype0
?
Adam/gsc_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *$
shared_nameAdam/gsc_9/kernel/m
?
'Adam/gsc_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gsc_9/kernel/m**
_output_shapes
:  *
dtype0
z
Adam/gsc_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/gsc_9/bias/m
s
%Adam/gsc_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/gsc_9/bias/m*
_output_shapes
: *
dtype0
?
Adam/opt_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *$
shared_nameAdam/opt_9/kernel/m
?
'Adam/opt_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/opt_9/kernel/m**
_output_shapes
:  *
dtype0
z
Adam/opt_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/opt_9/bias/m
s
%Adam/opt_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/opt_9/bias/m*
_output_shapes
: *
dtype0
?
Adam/gsc_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/gsc_11/kernel/m
?
(Adam/gsc_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gsc_11/kernel/m**
_output_shapes
:  *
dtype0
|
Adam/gsc_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/gsc_11/bias/m
u
&Adam/gsc_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/gsc_11/bias/m*
_output_shapes
: *
dtype0
?
Adam/opt_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/opt_11/kernel/m
?
(Adam/opt_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/opt_11/kernel/m**
_output_shapes
:  *
dtype0
|
Adam/opt_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/opt_11/bias/m
u
&Adam/opt_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/opt_11/bias/m*
_output_shapes
: *
dtype0
?
Adam/gsc_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/gsc_12/kernel/m
?
(Adam/gsc_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gsc_12/kernel/m**
_output_shapes
:  *
dtype0
|
Adam/gsc_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/gsc_12/bias/m
u
&Adam/gsc_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/gsc_12/bias/m*
_output_shapes
: *
dtype0
?
Adam/opt_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/opt_12/kernel/m
?
(Adam/opt_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/opt_12/kernel/m**
_output_shapes
:  *
dtype0
|
Adam/opt_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/opt_12/bias/m
u
&Adam/opt_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/opt_12/bias/m*
_output_shapes
: *
dtype0
?
Adam/merge_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*&
shared_nameAdam/merge_1/kernel/m
?
)Adam/merge_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/merge_1/kernel/m**
_output_shapes
: @*
dtype0
~
Adam/merge_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/merge_1/bias/m
w
'Adam/merge_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/merge_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/merge_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_nameAdam/merge_2/kernel/m
?
)Adam/merge_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/merge_2/kernel/m**
_output_shapes
:@@*
dtype0
~
Adam/merge_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/merge_2/bias/m
w
'Adam/merge_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/merge_2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/merge_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_nameAdam/merge_4/kernel/m
?
)Adam/merge_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/merge_4/kernel/m**
_output_shapes
:@@*
dtype0
~
Adam/merge_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/merge_4/bias/m
w
'Adam/merge_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/merge_4/bias/m*
_output_shapes
:@*
dtype0
?
Adam/merge_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_nameAdam/merge_5/kernel/m
?
)Adam/merge_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/merge_5/kernel/m**
_output_shapes
:@@*
dtype0
~
Adam/merge_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/merge_5/bias/m
w
'Adam/merge_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/merge_5/bias/m*
_output_shapes
:@*
dtype0
?
Adam/merge_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?*&
shared_nameAdam/merge_7/kernel/m
?
)Adam/merge_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/merge_7/kernel/m*+
_output_shapes
:@?*
dtype0

Adam/merge_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/merge_7/bias/m
x
'Adam/merge_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/merge_7/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/merge_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??*&
shared_nameAdam/merge_8/kernel/m
?
)Adam/merge_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/merge_8/kernel/m*,
_output_shapes
:??*
dtype0

Adam/merge_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/merge_8/bias/m
x
'Adam/merge_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/merge_8/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/fc_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/fc_1/kernel/m
{
&Adam/fc_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc_1/kernel/m* 
_output_shapes
:
??*
dtype0
y
Adam/fc_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/fc_1/bias/m
r
$Adam/fc_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/fc_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/fc_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *#
shared_nameAdam/fc_3/kernel/m
z
&Adam/fc_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc_3/kernel/m*
_output_shapes
:	? *
dtype0
x
Adam/fc_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/fc_3/bias/m
q
$Adam/fc_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/fc_3/bias/m*
_output_shapes
: *
dtype0
?
Adam/pred/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *#
shared_nameAdam/pred/kernel/m
y
&Adam/pred/kernel/m/Read/ReadVariableOpReadVariableOpAdam/pred/kernel/m*
_output_shapes

: *
dtype0
x
Adam/pred/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/pred/bias/m
q
$Adam/pred/bias/m/Read/ReadVariableOpReadVariableOpAdam/pred/bias/m*
_output_shapes
:*
dtype0
?
Adam/gsc_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/gsc_2/kernel/v
?
'Adam/gsc_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gsc_2/kernel/v**
_output_shapes
:*
dtype0
z
Adam/gsc_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/gsc_2/bias/v
s
%Adam/gsc_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/gsc_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/opt_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/opt_2/kernel/v
?
'Adam/opt_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/opt_2/kernel/v**
_output_shapes
:*
dtype0
z
Adam/opt_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/opt_2/bias/v
s
%Adam/opt_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/opt_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/gsc_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/gsc_3/kernel/v
?
'Adam/gsc_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gsc_3/kernel/v**
_output_shapes
:*
dtype0
z
Adam/gsc_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/gsc_3/bias/v
s
%Adam/gsc_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/gsc_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/opt_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/opt_3/kernel/v
?
'Adam/opt_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/opt_3/kernel/v**
_output_shapes
:*
dtype0
z
Adam/opt_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/opt_3/bias/v
s
%Adam/opt_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/opt_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/gsc_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/gsc_5/kernel/v
?
'Adam/gsc_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gsc_5/kernel/v**
_output_shapes
:*
dtype0
z
Adam/gsc_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/gsc_5/bias/v
s
%Adam/gsc_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/gsc_5/bias/v*
_output_shapes
:*
dtype0
?
Adam/opt_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/opt_5/kernel/v
?
'Adam/opt_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/opt_5/kernel/v**
_output_shapes
:*
dtype0
z
Adam/opt_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/opt_5/bias/v
s
%Adam/opt_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/opt_5/bias/v*
_output_shapes
:*
dtype0
?
Adam/gsc_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/gsc_6/kernel/v
?
'Adam/gsc_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gsc_6/kernel/v**
_output_shapes
:*
dtype0
z
Adam/gsc_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/gsc_6/bias/v
s
%Adam/gsc_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/gsc_6/bias/v*
_output_shapes
:*
dtype0
?
Adam/opt_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/opt_6/kernel/v
?
'Adam/opt_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/opt_6/kernel/v**
_output_shapes
:*
dtype0
z
Adam/opt_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/opt_6/bias/v
s
%Adam/opt_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/opt_6/bias/v*
_output_shapes
:*
dtype0
?
Adam/gsc_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/gsc_8/kernel/v
?
'Adam/gsc_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gsc_8/kernel/v**
_output_shapes
: *
dtype0
z
Adam/gsc_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/gsc_8/bias/v
s
%Adam/gsc_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/gsc_8/bias/v*
_output_shapes
: *
dtype0
?
Adam/opt_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/opt_8/kernel/v
?
'Adam/opt_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/opt_8/kernel/v**
_output_shapes
: *
dtype0
z
Adam/opt_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/opt_8/bias/v
s
%Adam/opt_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/opt_8/bias/v*
_output_shapes
: *
dtype0
?
Adam/gsc_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *$
shared_nameAdam/gsc_9/kernel/v
?
'Adam/gsc_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gsc_9/kernel/v**
_output_shapes
:  *
dtype0
z
Adam/gsc_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/gsc_9/bias/v
s
%Adam/gsc_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/gsc_9/bias/v*
_output_shapes
: *
dtype0
?
Adam/opt_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *$
shared_nameAdam/opt_9/kernel/v
?
'Adam/opt_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/opt_9/kernel/v**
_output_shapes
:  *
dtype0
z
Adam/opt_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/opt_9/bias/v
s
%Adam/opt_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/opt_9/bias/v*
_output_shapes
: *
dtype0
?
Adam/gsc_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/gsc_11/kernel/v
?
(Adam/gsc_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gsc_11/kernel/v**
_output_shapes
:  *
dtype0
|
Adam/gsc_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/gsc_11/bias/v
u
&Adam/gsc_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/gsc_11/bias/v*
_output_shapes
: *
dtype0
?
Adam/opt_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/opt_11/kernel/v
?
(Adam/opt_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/opt_11/kernel/v**
_output_shapes
:  *
dtype0
|
Adam/opt_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/opt_11/bias/v
u
&Adam/opt_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/opt_11/bias/v*
_output_shapes
: *
dtype0
?
Adam/gsc_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/gsc_12/kernel/v
?
(Adam/gsc_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gsc_12/kernel/v**
_output_shapes
:  *
dtype0
|
Adam/gsc_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/gsc_12/bias/v
u
&Adam/gsc_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/gsc_12/bias/v*
_output_shapes
: *
dtype0
?
Adam/opt_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/opt_12/kernel/v
?
(Adam/opt_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/opt_12/kernel/v**
_output_shapes
:  *
dtype0
|
Adam/opt_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/opt_12/bias/v
u
&Adam/opt_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/opt_12/bias/v*
_output_shapes
: *
dtype0
?
Adam/merge_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*&
shared_nameAdam/merge_1/kernel/v
?
)Adam/merge_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/merge_1/kernel/v**
_output_shapes
: @*
dtype0
~
Adam/merge_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/merge_1/bias/v
w
'Adam/merge_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/merge_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/merge_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_nameAdam/merge_2/kernel/v
?
)Adam/merge_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/merge_2/kernel/v**
_output_shapes
:@@*
dtype0
~
Adam/merge_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/merge_2/bias/v
w
'Adam/merge_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/merge_2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/merge_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_nameAdam/merge_4/kernel/v
?
)Adam/merge_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/merge_4/kernel/v**
_output_shapes
:@@*
dtype0
~
Adam/merge_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/merge_4/bias/v
w
'Adam/merge_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/merge_4/bias/v*
_output_shapes
:@*
dtype0
?
Adam/merge_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*&
shared_nameAdam/merge_5/kernel/v
?
)Adam/merge_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/merge_5/kernel/v**
_output_shapes
:@@*
dtype0
~
Adam/merge_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/merge_5/bias/v
w
'Adam/merge_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/merge_5/bias/v*
_output_shapes
:@*
dtype0
?
Adam/merge_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@?*&
shared_nameAdam/merge_7/kernel/v
?
)Adam/merge_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/merge_7/kernel/v*+
_output_shapes
:@?*
dtype0

Adam/merge_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/merge_7/bias/v
x
'Adam/merge_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/merge_7/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/merge_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*!
shape:??*&
shared_nameAdam/merge_8/kernel/v
?
)Adam/merge_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/merge_8/kernel/v*,
_output_shapes
:??*
dtype0

Adam/merge_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/merge_8/bias/v
x
'Adam/merge_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/merge_8/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/fc_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameAdam/fc_1/kernel/v
{
&Adam/fc_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc_1/kernel/v* 
_output_shapes
:
??*
dtype0
y
Adam/fc_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameAdam/fc_1/bias/v
r
$Adam/fc_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/fc_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/fc_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *#
shared_nameAdam/fc_3/kernel/v
z
&Adam/fc_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc_3/kernel/v*
_output_shapes
:	? *
dtype0
x
Adam/fc_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/fc_3/bias/v
q
$Adam/fc_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/fc_3/bias/v*
_output_shapes
: *
dtype0
?
Adam/pred/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *#
shared_nameAdam/pred/kernel/v
y
&Adam/pred/kernel/v/Read/ReadVariableOpReadVariableOpAdam/pred/kernel/v*
_output_shapes

: *
dtype0
x
Adam/pred/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/pred/bias/v
q
$Adam/pred/bias/v/Read/ReadVariableOpReadVariableOpAdam/pred/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer-20
layer_with_weights-12
layer-21
layer_with_weights-13
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer-25
layer-26
layer-27
layer-28
layer_with_weights-16
layer-29
layer_with_weights-17
layer-30
 layer-31
!layer_with_weights-18
!layer-32
"layer_with_weights-19
"layer-33
#layer-34
$layer_with_weights-20
$layer-35
%layer_with_weights-21
%layer-36
&layer-37
'layer-38
(layer_with_weights-22
(layer-39
)layer-40
*layer_with_weights-23
*layer-41
+layer_with_weights-24
+layer-42
,	optimizer
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3_default_save_signature
4
signatures*
* 
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses* 
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses* 
?

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses*
?

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses*
?

Qkernel
Rbias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses*
?

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*
?
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
?
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses* 
?

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses*
?

ukernel
vbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses*
?

}kernel
~bias
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateAm?Bm?Im?Jm?Qm?Rm?Ym?Zm?mm?nm?um?vm?}m?~m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Av?Bv?Iv?Jv?Qv?Rv?Yv?Zv?mv?nv?uv?vv?}v?~v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?*
?
A0
B1
I2
J3
Q4
R5
Y6
Z7
m8
n9
u10
v11
}12
~13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49*
?
A0
B1
I2
J3
Q4
R5
Y6
Z7
m8
n9
u10
v11
}12
~13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
3_default_save_signature
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*
* 
* 
* 

?serving_default* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEgsc_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
gsc_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1*

A0
B1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEopt_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
opt_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEgsc_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
gsc_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

Q0
R1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEopt_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
opt_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEgsc_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
gsc_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

m0
n1*

m0
n1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEopt_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
opt_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

u0
v1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEgsc_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
gsc_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

}0
~1*

}0
~1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEopt_6/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
opt_6/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEgsc_8/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
gsc_8/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEopt_8/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
opt_8/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEgsc_9/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
gsc_9/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEopt_9/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
opt_9/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEgsc_11/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEgsc_11/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEopt_11/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEopt_11/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEgsc_12/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEgsc_12/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEopt_12/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEopt_12/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEmerge_1/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmerge_1/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEmerge_2/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmerge_2/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEmerge_4/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmerge_4/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEmerge_5/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmerge_5/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEmerge_7/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmerge_7/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEmerge_8/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmerge_8/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEfc_1/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE	fc_1/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
\V
VARIABLE_VALUEfc_3/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE	fc_3/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEpred/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE	pred/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42*
$
?0
?1
?2
?3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
`
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api*
`
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
* 
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
y
VARIABLE_VALUEAdam/gsc_2/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gsc_2/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/opt_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/opt_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/gsc_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gsc_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/opt_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/opt_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/gsc_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gsc_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/opt_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/opt_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/gsc_6/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gsc_6/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/opt_6/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/opt_6/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/gsc_8/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gsc_8/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/opt_8/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/opt_8/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/gsc_9/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/gsc_9/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/opt_9/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/opt_9/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/gsc_11/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/gsc_11/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/opt_11/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/opt_11/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/gsc_12/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/gsc_12/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/opt_12/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/opt_12/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_1/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_1/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_2/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_2/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_4/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_4/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_5/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_5/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_7/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_7/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_8/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_8/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/fc_1/kernel/mSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/fc_1/bias/mQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/fc_3/kernel/mSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/fc_3/bias/mQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/pred/kernel/mSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/pred/bias/mQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/gsc_2/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gsc_2/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/opt_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/opt_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/gsc_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gsc_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/opt_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/opt_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/gsc_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gsc_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/opt_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/opt_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/gsc_6/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gsc_6/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/opt_6/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/opt_6/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/gsc_8/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gsc_8/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/opt_8/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/opt_8/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/gsc_9/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/gsc_9/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/opt_9/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/opt_9/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/gsc_11/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/gsc_11/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/opt_11/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/opt_11/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/gsc_12/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/gsc_12/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/opt_12/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/opt_12/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_1/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_1/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_2/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_2/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_4/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_4/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_5/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_5/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_7/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_7/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/merge_8/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/merge_8/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/fc_1/kernel/vSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/fc_1/bias/vQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/fc_3/kernel/vSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/fc_3/bias/vQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/pred/kernel/vSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/pred/bias/vQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_1Placeholder*5
_output_shapes#
!:?????????@??*
dtype0**
shape!:?????????@??
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1opt_2/kernel
opt_2/biasgsc_2/kernel
gsc_2/biasopt_3/kernel
opt_3/biasgsc_3/kernel
gsc_3/biasopt_5/kernel
opt_5/biasgsc_5/kernel
gsc_5/biasopt_6/kernel
opt_6/biasgsc_6/kernel
gsc_6/biasopt_8/kernel
opt_8/biasgsc_8/kernel
gsc_8/biasopt_9/kernel
opt_9/biasgsc_9/kernel
gsc_9/biasopt_11/kernelopt_11/biasgsc_11/kernelgsc_11/biasopt_12/kernelopt_12/biasgsc_12/kernelgsc_12/biasmerge_1/kernelmerge_1/biasmerge_2/kernelmerge_2/biasmerge_4/kernelmerge_4/biasmerge_5/kernelmerge_5/biasmerge_7/kernelmerge_7/biasmerge_8/kernelmerge_8/biasfc_1/kernel	fc_1/biasfc_3/kernel	fc_3/biaspred/kernel	pred/bias*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_71269
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?3
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename gsc_2/kernel/Read/ReadVariableOpgsc_2/bias/Read/ReadVariableOp opt_2/kernel/Read/ReadVariableOpopt_2/bias/Read/ReadVariableOp gsc_3/kernel/Read/ReadVariableOpgsc_3/bias/Read/ReadVariableOp opt_3/kernel/Read/ReadVariableOpopt_3/bias/Read/ReadVariableOp gsc_5/kernel/Read/ReadVariableOpgsc_5/bias/Read/ReadVariableOp opt_5/kernel/Read/ReadVariableOpopt_5/bias/Read/ReadVariableOp gsc_6/kernel/Read/ReadVariableOpgsc_6/bias/Read/ReadVariableOp opt_6/kernel/Read/ReadVariableOpopt_6/bias/Read/ReadVariableOp gsc_8/kernel/Read/ReadVariableOpgsc_8/bias/Read/ReadVariableOp opt_8/kernel/Read/ReadVariableOpopt_8/bias/Read/ReadVariableOp gsc_9/kernel/Read/ReadVariableOpgsc_9/bias/Read/ReadVariableOp opt_9/kernel/Read/ReadVariableOpopt_9/bias/Read/ReadVariableOp!gsc_11/kernel/Read/ReadVariableOpgsc_11/bias/Read/ReadVariableOp!opt_11/kernel/Read/ReadVariableOpopt_11/bias/Read/ReadVariableOp!gsc_12/kernel/Read/ReadVariableOpgsc_12/bias/Read/ReadVariableOp!opt_12/kernel/Read/ReadVariableOpopt_12/bias/Read/ReadVariableOp"merge_1/kernel/Read/ReadVariableOp merge_1/bias/Read/ReadVariableOp"merge_2/kernel/Read/ReadVariableOp merge_2/bias/Read/ReadVariableOp"merge_4/kernel/Read/ReadVariableOp merge_4/bias/Read/ReadVariableOp"merge_5/kernel/Read/ReadVariableOp merge_5/bias/Read/ReadVariableOp"merge_7/kernel/Read/ReadVariableOp merge_7/bias/Read/ReadVariableOp"merge_8/kernel/Read/ReadVariableOp merge_8/bias/Read/ReadVariableOpfc_1/kernel/Read/ReadVariableOpfc_1/bias/Read/ReadVariableOpfc_3/kernel/Read/ReadVariableOpfc_3/bias/Read/ReadVariableOppred/kernel/Read/ReadVariableOppred/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/gsc_2/kernel/m/Read/ReadVariableOp%Adam/gsc_2/bias/m/Read/ReadVariableOp'Adam/opt_2/kernel/m/Read/ReadVariableOp%Adam/opt_2/bias/m/Read/ReadVariableOp'Adam/gsc_3/kernel/m/Read/ReadVariableOp%Adam/gsc_3/bias/m/Read/ReadVariableOp'Adam/opt_3/kernel/m/Read/ReadVariableOp%Adam/opt_3/bias/m/Read/ReadVariableOp'Adam/gsc_5/kernel/m/Read/ReadVariableOp%Adam/gsc_5/bias/m/Read/ReadVariableOp'Adam/opt_5/kernel/m/Read/ReadVariableOp%Adam/opt_5/bias/m/Read/ReadVariableOp'Adam/gsc_6/kernel/m/Read/ReadVariableOp%Adam/gsc_6/bias/m/Read/ReadVariableOp'Adam/opt_6/kernel/m/Read/ReadVariableOp%Adam/opt_6/bias/m/Read/ReadVariableOp'Adam/gsc_8/kernel/m/Read/ReadVariableOp%Adam/gsc_8/bias/m/Read/ReadVariableOp'Adam/opt_8/kernel/m/Read/ReadVariableOp%Adam/opt_8/bias/m/Read/ReadVariableOp'Adam/gsc_9/kernel/m/Read/ReadVariableOp%Adam/gsc_9/bias/m/Read/ReadVariableOp'Adam/opt_9/kernel/m/Read/ReadVariableOp%Adam/opt_9/bias/m/Read/ReadVariableOp(Adam/gsc_11/kernel/m/Read/ReadVariableOp&Adam/gsc_11/bias/m/Read/ReadVariableOp(Adam/opt_11/kernel/m/Read/ReadVariableOp&Adam/opt_11/bias/m/Read/ReadVariableOp(Adam/gsc_12/kernel/m/Read/ReadVariableOp&Adam/gsc_12/bias/m/Read/ReadVariableOp(Adam/opt_12/kernel/m/Read/ReadVariableOp&Adam/opt_12/bias/m/Read/ReadVariableOp)Adam/merge_1/kernel/m/Read/ReadVariableOp'Adam/merge_1/bias/m/Read/ReadVariableOp)Adam/merge_2/kernel/m/Read/ReadVariableOp'Adam/merge_2/bias/m/Read/ReadVariableOp)Adam/merge_4/kernel/m/Read/ReadVariableOp'Adam/merge_4/bias/m/Read/ReadVariableOp)Adam/merge_5/kernel/m/Read/ReadVariableOp'Adam/merge_5/bias/m/Read/ReadVariableOp)Adam/merge_7/kernel/m/Read/ReadVariableOp'Adam/merge_7/bias/m/Read/ReadVariableOp)Adam/merge_8/kernel/m/Read/ReadVariableOp'Adam/merge_8/bias/m/Read/ReadVariableOp&Adam/fc_1/kernel/m/Read/ReadVariableOp$Adam/fc_1/bias/m/Read/ReadVariableOp&Adam/fc_3/kernel/m/Read/ReadVariableOp$Adam/fc_3/bias/m/Read/ReadVariableOp&Adam/pred/kernel/m/Read/ReadVariableOp$Adam/pred/bias/m/Read/ReadVariableOp'Adam/gsc_2/kernel/v/Read/ReadVariableOp%Adam/gsc_2/bias/v/Read/ReadVariableOp'Adam/opt_2/kernel/v/Read/ReadVariableOp%Adam/opt_2/bias/v/Read/ReadVariableOp'Adam/gsc_3/kernel/v/Read/ReadVariableOp%Adam/gsc_3/bias/v/Read/ReadVariableOp'Adam/opt_3/kernel/v/Read/ReadVariableOp%Adam/opt_3/bias/v/Read/ReadVariableOp'Adam/gsc_5/kernel/v/Read/ReadVariableOp%Adam/gsc_5/bias/v/Read/ReadVariableOp'Adam/opt_5/kernel/v/Read/ReadVariableOp%Adam/opt_5/bias/v/Read/ReadVariableOp'Adam/gsc_6/kernel/v/Read/ReadVariableOp%Adam/gsc_6/bias/v/Read/ReadVariableOp'Adam/opt_6/kernel/v/Read/ReadVariableOp%Adam/opt_6/bias/v/Read/ReadVariableOp'Adam/gsc_8/kernel/v/Read/ReadVariableOp%Adam/gsc_8/bias/v/Read/ReadVariableOp'Adam/opt_8/kernel/v/Read/ReadVariableOp%Adam/opt_8/bias/v/Read/ReadVariableOp'Adam/gsc_9/kernel/v/Read/ReadVariableOp%Adam/gsc_9/bias/v/Read/ReadVariableOp'Adam/opt_9/kernel/v/Read/ReadVariableOp%Adam/opt_9/bias/v/Read/ReadVariableOp(Adam/gsc_11/kernel/v/Read/ReadVariableOp&Adam/gsc_11/bias/v/Read/ReadVariableOp(Adam/opt_11/kernel/v/Read/ReadVariableOp&Adam/opt_11/bias/v/Read/ReadVariableOp(Adam/gsc_12/kernel/v/Read/ReadVariableOp&Adam/gsc_12/bias/v/Read/ReadVariableOp(Adam/opt_12/kernel/v/Read/ReadVariableOp&Adam/opt_12/bias/v/Read/ReadVariableOp)Adam/merge_1/kernel/v/Read/ReadVariableOp'Adam/merge_1/bias/v/Read/ReadVariableOp)Adam/merge_2/kernel/v/Read/ReadVariableOp'Adam/merge_2/bias/v/Read/ReadVariableOp)Adam/merge_4/kernel/v/Read/ReadVariableOp'Adam/merge_4/bias/v/Read/ReadVariableOp)Adam/merge_5/kernel/v/Read/ReadVariableOp'Adam/merge_5/bias/v/Read/ReadVariableOp)Adam/merge_7/kernel/v/Read/ReadVariableOp'Adam/merge_7/bias/v/Read/ReadVariableOp)Adam/merge_8/kernel/v/Read/ReadVariableOp'Adam/merge_8/bias/v/Read/ReadVariableOp&Adam/fc_1/kernel/v/Read/ReadVariableOp$Adam/fc_1/bias/v/Read/ReadVariableOp&Adam/fc_3/kernel/v/Read/ReadVariableOp$Adam/fc_3/bias/v/Read/ReadVariableOp&Adam/pred/kernel/v/Read/ReadVariableOp$Adam/pred/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_72503
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegsc_2/kernel
gsc_2/biasopt_2/kernel
opt_2/biasgsc_3/kernel
gsc_3/biasopt_3/kernel
opt_3/biasgsc_5/kernel
gsc_5/biasopt_5/kernel
opt_5/biasgsc_6/kernel
gsc_6/biasopt_6/kernel
opt_6/biasgsc_8/kernel
gsc_8/biasopt_8/kernel
opt_8/biasgsc_9/kernel
gsc_9/biasopt_9/kernel
opt_9/biasgsc_11/kernelgsc_11/biasopt_11/kernelopt_11/biasgsc_12/kernelgsc_12/biasopt_12/kernelopt_12/biasmerge_1/kernelmerge_1/biasmerge_2/kernelmerge_2/biasmerge_4/kernelmerge_4/biasmerge_5/kernelmerge_5/biasmerge_7/kernelmerge_7/biasmerge_8/kernelmerge_8/biasfc_1/kernel	fc_1/biasfc_3/kernel	fc_3/biaspred/kernel	pred/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivesfalse_positivestrue_positives_1false_negativestotal_1count_1Adam/gsc_2/kernel/mAdam/gsc_2/bias/mAdam/opt_2/kernel/mAdam/opt_2/bias/mAdam/gsc_3/kernel/mAdam/gsc_3/bias/mAdam/opt_3/kernel/mAdam/opt_3/bias/mAdam/gsc_5/kernel/mAdam/gsc_5/bias/mAdam/opt_5/kernel/mAdam/opt_5/bias/mAdam/gsc_6/kernel/mAdam/gsc_6/bias/mAdam/opt_6/kernel/mAdam/opt_6/bias/mAdam/gsc_8/kernel/mAdam/gsc_8/bias/mAdam/opt_8/kernel/mAdam/opt_8/bias/mAdam/gsc_9/kernel/mAdam/gsc_9/bias/mAdam/opt_9/kernel/mAdam/opt_9/bias/mAdam/gsc_11/kernel/mAdam/gsc_11/bias/mAdam/opt_11/kernel/mAdam/opt_11/bias/mAdam/gsc_12/kernel/mAdam/gsc_12/bias/mAdam/opt_12/kernel/mAdam/opt_12/bias/mAdam/merge_1/kernel/mAdam/merge_1/bias/mAdam/merge_2/kernel/mAdam/merge_2/bias/mAdam/merge_4/kernel/mAdam/merge_4/bias/mAdam/merge_5/kernel/mAdam/merge_5/bias/mAdam/merge_7/kernel/mAdam/merge_7/bias/mAdam/merge_8/kernel/mAdam/merge_8/bias/mAdam/fc_1/kernel/mAdam/fc_1/bias/mAdam/fc_3/kernel/mAdam/fc_3/bias/mAdam/pred/kernel/mAdam/pred/bias/mAdam/gsc_2/kernel/vAdam/gsc_2/bias/vAdam/opt_2/kernel/vAdam/opt_2/bias/vAdam/gsc_3/kernel/vAdam/gsc_3/bias/vAdam/opt_3/kernel/vAdam/opt_3/bias/vAdam/gsc_5/kernel/vAdam/gsc_5/bias/vAdam/opt_5/kernel/vAdam/opt_5/bias/vAdam/gsc_6/kernel/vAdam/gsc_6/bias/vAdam/opt_6/kernel/vAdam/opt_6/bias/vAdam/gsc_8/kernel/vAdam/gsc_8/bias/vAdam/opt_8/kernel/vAdam/opt_8/bias/vAdam/gsc_9/kernel/vAdam/gsc_9/bias/vAdam/opt_9/kernel/vAdam/opt_9/bias/vAdam/gsc_11/kernel/vAdam/gsc_11/bias/vAdam/opt_11/kernel/vAdam/opt_11/bias/vAdam/gsc_12/kernel/vAdam/gsc_12/bias/vAdam/opt_12/kernel/vAdam/opt_12/bias/vAdam/merge_1/kernel/vAdam/merge_1/bias/vAdam/merge_2/kernel/vAdam/merge_2/bias/vAdam/merge_4/kernel/vAdam/merge_4/bias/vAdam/merge_5/kernel/vAdam/merge_5/bias/vAdam/merge_7/kernel/vAdam/merge_7/bias/vAdam/merge_8/kernel/vAdam/merge_8/bias/vAdam/fc_1/kernel/vAdam/fc_1/bias/vAdam/fc_3/kernel/vAdam/fc_3/bias/vAdam/pred/kernel/vAdam/pred/bias/v*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_73002??
?
\
@__inference_opt_1_layer_call_and_return_conditional_losses_71313

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*
ellipsis_mask*
end_maskl
IdentityIdentitystrided_slice:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
B
&__inference_opt_10_layer_call_fn_71616

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_10_layer_call_and_return_conditional_losses_68790?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_fuse_2_layer_call_and_return_conditional_losses_71743

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_merge_1_layer_call_and_return_conditional_losses_71763

inputs<
conv3d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:????????? 
 
_user_specified_nameinputs
?
?
@__inference_opt_3_layer_call_and_return_conditional_losses_71401

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@??o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:?????????@??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
%__inference_gsc_8_layer_call_fn_71530

inputs%
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_8_layer_call_and_return_conditional_losses_69060{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@88 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@88
 
_user_specified_nameinputs
??
?g
!__inference__traced_restore_73002
file_prefix;
assignvariableop_gsc_2_kernel:+
assignvariableop_1_gsc_2_bias:=
assignvariableop_2_opt_2_kernel:+
assignvariableop_3_opt_2_bias:=
assignvariableop_4_gsc_3_kernel:+
assignvariableop_5_gsc_3_bias:=
assignvariableop_6_opt_3_kernel:+
assignvariableop_7_opt_3_bias:=
assignvariableop_8_gsc_5_kernel:+
assignvariableop_9_gsc_5_bias:>
 assignvariableop_10_opt_5_kernel:,
assignvariableop_11_opt_5_bias:>
 assignvariableop_12_gsc_6_kernel:,
assignvariableop_13_gsc_6_bias:>
 assignvariableop_14_opt_6_kernel:,
assignvariableop_15_opt_6_bias:>
 assignvariableop_16_gsc_8_kernel: ,
assignvariableop_17_gsc_8_bias: >
 assignvariableop_18_opt_8_kernel: ,
assignvariableop_19_opt_8_bias: >
 assignvariableop_20_gsc_9_kernel:  ,
assignvariableop_21_gsc_9_bias: >
 assignvariableop_22_opt_9_kernel:  ,
assignvariableop_23_opt_9_bias: ?
!assignvariableop_24_gsc_11_kernel:  -
assignvariableop_25_gsc_11_bias: ?
!assignvariableop_26_opt_11_kernel:  -
assignvariableop_27_opt_11_bias: ?
!assignvariableop_28_gsc_12_kernel:  -
assignvariableop_29_gsc_12_bias: ?
!assignvariableop_30_opt_12_kernel:  -
assignvariableop_31_opt_12_bias: @
"assignvariableop_32_merge_1_kernel: @.
 assignvariableop_33_merge_1_bias:@@
"assignvariableop_34_merge_2_kernel:@@.
 assignvariableop_35_merge_2_bias:@@
"assignvariableop_36_merge_4_kernel:@@.
 assignvariableop_37_merge_4_bias:@@
"assignvariableop_38_merge_5_kernel:@@.
 assignvariableop_39_merge_5_bias:@A
"assignvariableop_40_merge_7_kernel:@?/
 assignvariableop_41_merge_7_bias:	?B
"assignvariableop_42_merge_8_kernel:??/
 assignvariableop_43_merge_8_bias:	?3
assignvariableop_44_fc_1_kernel:
??,
assignvariableop_45_fc_1_bias:	?2
assignvariableop_46_fc_3_kernel:	? +
assignvariableop_47_fc_3_bias: 1
assignvariableop_48_pred_kernel: +
assignvariableop_49_pred_bias:'
assignvariableop_50_adam_iter:	 )
assignvariableop_51_adam_beta_1: )
assignvariableop_52_adam_beta_2: (
assignvariableop_53_adam_decay: 0
&assignvariableop_54_adam_learning_rate: #
assignvariableop_55_total: #
assignvariableop_56_count: 0
"assignvariableop_57_true_positives:1
#assignvariableop_58_false_positives:2
$assignvariableop_59_true_positives_1:1
#assignvariableop_60_false_negatives:%
assignvariableop_61_total_1: %
assignvariableop_62_count_1: E
'assignvariableop_63_adam_gsc_2_kernel_m:3
%assignvariableop_64_adam_gsc_2_bias_m:E
'assignvariableop_65_adam_opt_2_kernel_m:3
%assignvariableop_66_adam_opt_2_bias_m:E
'assignvariableop_67_adam_gsc_3_kernel_m:3
%assignvariableop_68_adam_gsc_3_bias_m:E
'assignvariableop_69_adam_opt_3_kernel_m:3
%assignvariableop_70_adam_opt_3_bias_m:E
'assignvariableop_71_adam_gsc_5_kernel_m:3
%assignvariableop_72_adam_gsc_5_bias_m:E
'assignvariableop_73_adam_opt_5_kernel_m:3
%assignvariableop_74_adam_opt_5_bias_m:E
'assignvariableop_75_adam_gsc_6_kernel_m:3
%assignvariableop_76_adam_gsc_6_bias_m:E
'assignvariableop_77_adam_opt_6_kernel_m:3
%assignvariableop_78_adam_opt_6_bias_m:E
'assignvariableop_79_adam_gsc_8_kernel_m: 3
%assignvariableop_80_adam_gsc_8_bias_m: E
'assignvariableop_81_adam_opt_8_kernel_m: 3
%assignvariableop_82_adam_opt_8_bias_m: E
'assignvariableop_83_adam_gsc_9_kernel_m:  3
%assignvariableop_84_adam_gsc_9_bias_m: E
'assignvariableop_85_adam_opt_9_kernel_m:  3
%assignvariableop_86_adam_opt_9_bias_m: F
(assignvariableop_87_adam_gsc_11_kernel_m:  4
&assignvariableop_88_adam_gsc_11_bias_m: F
(assignvariableop_89_adam_opt_11_kernel_m:  4
&assignvariableop_90_adam_opt_11_bias_m: F
(assignvariableop_91_adam_gsc_12_kernel_m:  4
&assignvariableop_92_adam_gsc_12_bias_m: F
(assignvariableop_93_adam_opt_12_kernel_m:  4
&assignvariableop_94_adam_opt_12_bias_m: G
)assignvariableop_95_adam_merge_1_kernel_m: @5
'assignvariableop_96_adam_merge_1_bias_m:@G
)assignvariableop_97_adam_merge_2_kernel_m:@@5
'assignvariableop_98_adam_merge_2_bias_m:@G
)assignvariableop_99_adam_merge_4_kernel_m:@@6
(assignvariableop_100_adam_merge_4_bias_m:@H
*assignvariableop_101_adam_merge_5_kernel_m:@@6
(assignvariableop_102_adam_merge_5_bias_m:@I
*assignvariableop_103_adam_merge_7_kernel_m:@?7
(assignvariableop_104_adam_merge_7_bias_m:	?J
*assignvariableop_105_adam_merge_8_kernel_m:??7
(assignvariableop_106_adam_merge_8_bias_m:	?;
'assignvariableop_107_adam_fc_1_kernel_m:
??4
%assignvariableop_108_adam_fc_1_bias_m:	?:
'assignvariableop_109_adam_fc_3_kernel_m:	? 3
%assignvariableop_110_adam_fc_3_bias_m: 9
'assignvariableop_111_adam_pred_kernel_m: 3
%assignvariableop_112_adam_pred_bias_m:F
(assignvariableop_113_adam_gsc_2_kernel_v:4
&assignvariableop_114_adam_gsc_2_bias_v:F
(assignvariableop_115_adam_opt_2_kernel_v:4
&assignvariableop_116_adam_opt_2_bias_v:F
(assignvariableop_117_adam_gsc_3_kernel_v:4
&assignvariableop_118_adam_gsc_3_bias_v:F
(assignvariableop_119_adam_opt_3_kernel_v:4
&assignvariableop_120_adam_opt_3_bias_v:F
(assignvariableop_121_adam_gsc_5_kernel_v:4
&assignvariableop_122_adam_gsc_5_bias_v:F
(assignvariableop_123_adam_opt_5_kernel_v:4
&assignvariableop_124_adam_opt_5_bias_v:F
(assignvariableop_125_adam_gsc_6_kernel_v:4
&assignvariableop_126_adam_gsc_6_bias_v:F
(assignvariableop_127_adam_opt_6_kernel_v:4
&assignvariableop_128_adam_opt_6_bias_v:F
(assignvariableop_129_adam_gsc_8_kernel_v: 4
&assignvariableop_130_adam_gsc_8_bias_v: F
(assignvariableop_131_adam_opt_8_kernel_v: 4
&assignvariableop_132_adam_opt_8_bias_v: F
(assignvariableop_133_adam_gsc_9_kernel_v:  4
&assignvariableop_134_adam_gsc_9_bias_v: F
(assignvariableop_135_adam_opt_9_kernel_v:  4
&assignvariableop_136_adam_opt_9_bias_v: G
)assignvariableop_137_adam_gsc_11_kernel_v:  5
'assignvariableop_138_adam_gsc_11_bias_v: G
)assignvariableop_139_adam_opt_11_kernel_v:  5
'assignvariableop_140_adam_opt_11_bias_v: G
)assignvariableop_141_adam_gsc_12_kernel_v:  5
'assignvariableop_142_adam_gsc_12_bias_v: G
)assignvariableop_143_adam_opt_12_kernel_v:  5
'assignvariableop_144_adam_opt_12_bias_v: H
*assignvariableop_145_adam_merge_1_kernel_v: @6
(assignvariableop_146_adam_merge_1_bias_v:@H
*assignvariableop_147_adam_merge_2_kernel_v:@@6
(assignvariableop_148_adam_merge_2_bias_v:@H
*assignvariableop_149_adam_merge_4_kernel_v:@@6
(assignvariableop_150_adam_merge_4_bias_v:@H
*assignvariableop_151_adam_merge_5_kernel_v:@@6
(assignvariableop_152_adam_merge_5_bias_v:@I
*assignvariableop_153_adam_merge_7_kernel_v:@?7
(assignvariableop_154_adam_merge_7_bias_v:	?J
*assignvariableop_155_adam_merge_8_kernel_v:??7
(assignvariableop_156_adam_merge_8_bias_v:	?;
'assignvariableop_157_adam_fc_1_kernel_v:
??4
%assignvariableop_158_adam_fc_1_bias_v:	?:
'assignvariableop_159_adam_fc_3_kernel_v:	? 3
%assignvariableop_160_adam_fc_3_bias_v: 9
'assignvariableop_161_adam_pred_kernel_v: 3
%assignvariableop_162_adam_pred_bias_v:
identity_164??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_135?AssignVariableOp_136?AssignVariableOp_137?AssignVariableOp_138?AssignVariableOp_139?AssignVariableOp_14?AssignVariableOp_140?AssignVariableOp_141?AssignVariableOp_142?AssignVariableOp_143?AssignVariableOp_144?AssignVariableOp_145?AssignVariableOp_146?AssignVariableOp_147?AssignVariableOp_148?AssignVariableOp_149?AssignVariableOp_15?AssignVariableOp_150?AssignVariableOp_151?AssignVariableOp_152?AssignVariableOp_153?AssignVariableOp_154?AssignVariableOp_155?AssignVariableOp_156?AssignVariableOp_157?AssignVariableOp_158?AssignVariableOp_159?AssignVariableOp_16?AssignVariableOp_160?AssignVariableOp_161?AssignVariableOp_162?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?]
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?\
value?\B?\?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_gsc_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_gsc_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_opt_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_opt_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_gsc_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_gsc_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_opt_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_opt_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_gsc_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_gsc_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp assignvariableop_10_opt_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_opt_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp assignvariableop_12_gsc_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_gsc_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp assignvariableop_14_opt_6_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_opt_6_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp assignvariableop_16_gsc_8_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_gsc_8_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp assignvariableop_18_opt_8_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_opt_8_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp assignvariableop_20_gsc_9_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_gsc_9_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp assignvariableop_22_opt_9_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_opt_9_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp!assignvariableop_24_gsc_11_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_gsc_11_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp!assignvariableop_26_opt_11_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_opt_11_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp!assignvariableop_28_gsc_12_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_gsc_12_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp!assignvariableop_30_opt_12_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpassignvariableop_31_opt_12_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp"assignvariableop_32_merge_1_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp assignvariableop_33_merge_1_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp"assignvariableop_34_merge_2_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp assignvariableop_35_merge_2_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp"assignvariableop_36_merge_4_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp assignvariableop_37_merge_4_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp"assignvariableop_38_merge_5_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp assignvariableop_39_merge_5_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp"assignvariableop_40_merge_7_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp assignvariableop_41_merge_7_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp"assignvariableop_42_merge_8_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp assignvariableop_43_merge_8_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOpassignvariableop_44_fc_1_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpassignvariableop_45_fc_1_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOpassignvariableop_46_fc_3_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpassignvariableop_47_fc_3_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOpassignvariableop_48_pred_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOpassignvariableop_49_pred_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_50AssignVariableOpassignvariableop_50_adam_iterIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpassignvariableop_51_adam_beta_1Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOpassignvariableop_52_adam_beta_2Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOpassignvariableop_53_adam_decayIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_learning_rateIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpassignvariableop_55_totalIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpassignvariableop_56_countIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp"assignvariableop_57_true_positivesIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp#assignvariableop_58_false_positivesIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp$assignvariableop_59_true_positives_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp#assignvariableop_60_false_negativesIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOpassignvariableop_61_total_1Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOpassignvariableop_62_count_1Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp'assignvariableop_63_adam_gsc_2_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp%assignvariableop_64_adam_gsc_2_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_opt_2_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp%assignvariableop_66_adam_opt_2_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_gsc_3_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp%assignvariableop_68_adam_gsc_3_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp'assignvariableop_69_adam_opt_3_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp%assignvariableop_70_adam_opt_3_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp'assignvariableop_71_adam_gsc_5_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp%assignvariableop_72_adam_gsc_5_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adam_opt_5_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp%assignvariableop_74_adam_opt_5_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp'assignvariableop_75_adam_gsc_6_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp%assignvariableop_76_adam_gsc_6_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp'assignvariableop_77_adam_opt_6_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp%assignvariableop_78_adam_opt_6_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp'assignvariableop_79_adam_gsc_8_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp%assignvariableop_80_adam_gsc_8_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp'assignvariableop_81_adam_opt_8_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp%assignvariableop_82_adam_opt_8_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp'assignvariableop_83_adam_gsc_9_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp%assignvariableop_84_adam_gsc_9_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp'assignvariableop_85_adam_opt_9_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp%assignvariableop_86_adam_opt_9_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp(assignvariableop_87_adam_gsc_11_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp&assignvariableop_88_adam_gsc_11_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp(assignvariableop_89_adam_opt_11_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp&assignvariableop_90_adam_opt_11_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp(assignvariableop_91_adam_gsc_12_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp&assignvariableop_92_adam_gsc_12_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp(assignvariableop_93_adam_opt_12_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOp&assignvariableop_94_adam_opt_12_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOp)assignvariableop_95_adam_merge_1_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOp'assignvariableop_96_adam_merge_1_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOp)assignvariableop_97_adam_merge_2_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOp'assignvariableop_98_adam_merge_2_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOp)assignvariableop_99_adam_merge_4_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOp(assignvariableop_100_adam_merge_4_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOp*assignvariableop_101_adam_merge_5_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOp(assignvariableop_102_adam_merge_5_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOp*assignvariableop_103_adam_merge_7_kernel_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOp(assignvariableop_104_adam_merge_7_bias_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOp*assignvariableop_105_adam_merge_8_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOp(assignvariableop_106_adam_merge_8_bias_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOp'assignvariableop_107_adam_fc_1_kernel_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOp%assignvariableop_108_adam_fc_1_bias_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOp'assignvariableop_109_adam_fc_3_kernel_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOp%assignvariableop_110_adam_fc_3_bias_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOp'assignvariableop_111_adam_pred_kernel_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOp%assignvariableop_112_adam_pred_bias_mIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOp(assignvariableop_113_adam_gsc_2_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOp&assignvariableop_114_adam_gsc_2_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOp(assignvariableop_115_adam_opt_2_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOp&assignvariableop_116_adam_opt_2_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOp(assignvariableop_117_adam_gsc_3_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOp&assignvariableop_118_adam_gsc_3_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOp(assignvariableop_119_adam_opt_3_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_120AssignVariableOp&assignvariableop_120_adam_opt_3_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_121AssignVariableOp(assignvariableop_121_adam_gsc_5_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_122AssignVariableOp&assignvariableop_122_adam_gsc_5_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_123AssignVariableOp(assignvariableop_123_adam_opt_5_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_124AssignVariableOp&assignvariableop_124_adam_opt_5_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_125AssignVariableOp(assignvariableop_125_adam_gsc_6_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_126AssignVariableOp&assignvariableop_126_adam_gsc_6_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_127AssignVariableOp(assignvariableop_127_adam_opt_6_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_128AssignVariableOp&assignvariableop_128_adam_opt_6_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_129AssignVariableOp(assignvariableop_129_adam_gsc_8_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_130AssignVariableOp&assignvariableop_130_adam_gsc_8_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_131AssignVariableOp(assignvariableop_131_adam_opt_8_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_132AssignVariableOp&assignvariableop_132_adam_opt_8_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_133AssignVariableOp(assignvariableop_133_adam_gsc_9_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_134AssignVariableOp&assignvariableop_134_adam_gsc_9_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_135AssignVariableOp(assignvariableop_135_adam_opt_9_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_136AssignVariableOp&assignvariableop_136_adam_opt_9_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_137AssignVariableOp)assignvariableop_137_adam_gsc_11_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_138AssignVariableOp'assignvariableop_138_adam_gsc_11_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_139AssignVariableOp)assignvariableop_139_adam_opt_11_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_140AssignVariableOp'assignvariableop_140_adam_opt_11_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_141AssignVariableOp)assignvariableop_141_adam_gsc_12_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_142AssignVariableOp'assignvariableop_142_adam_gsc_12_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_143AssignVariableOp)assignvariableop_143_adam_opt_12_kernel_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_144AssignVariableOp'assignvariableop_144_adam_opt_12_bias_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_145AssignVariableOp*assignvariableop_145_adam_merge_1_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_146AssignVariableOp(assignvariableop_146_adam_merge_1_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_147AssignVariableOp*assignvariableop_147_adam_merge_2_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_148AssignVariableOp(assignvariableop_148_adam_merge_2_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_149AssignVariableOp*assignvariableop_149_adam_merge_4_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_150AssignVariableOp(assignvariableop_150_adam_merge_4_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_151AssignVariableOp*assignvariableop_151_adam_merge_5_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_152AssignVariableOp(assignvariableop_152_adam_merge_5_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_153AssignVariableOp*assignvariableop_153_adam_merge_7_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_154AssignVariableOp(assignvariableop_154_adam_merge_7_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_155AssignVariableOp*assignvariableop_155_adam_merge_8_kernel_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_156AssignVariableOp(assignvariableop_156_adam_merge_8_bias_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_157AssignVariableOp'assignvariableop_157_adam_fc_1_kernel_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_158AssignVariableOp%assignvariableop_158_adam_fc_1_bias_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_159AssignVariableOp'assignvariableop_159_adam_fc_3_kernel_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_160AssignVariableOp%assignvariableop_160_adam_fc_3_bias_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_161AssignVariableOp'assignvariableop_161_adam_pred_kernel_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_162AssignVariableOp%assignvariableop_162_adam_pred_bias_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_163Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_164IdentityIdentity_163:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_164Identity_164:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
?__inference_fc_3_layer_call_and_return_conditional_losses_71971

inputs1
matmul_readvariableop_resource:	? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
k
A__inference_fuse_1_layer_call_and_return_conditional_losses_69178

inputs
inputs_1
identityZ
mulMulinputsinputs_1*
T0*3
_output_shapes!
:?????????@ [
IdentityIdentitymul:z:0*
T0*3
_output_shapes!
:?????????@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????@ :?????????@ :[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs:[W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
??
?>
__inference__traced_save_72503
file_prefix+
'savev2_gsc_2_kernel_read_readvariableop)
%savev2_gsc_2_bias_read_readvariableop+
'savev2_opt_2_kernel_read_readvariableop)
%savev2_opt_2_bias_read_readvariableop+
'savev2_gsc_3_kernel_read_readvariableop)
%savev2_gsc_3_bias_read_readvariableop+
'savev2_opt_3_kernel_read_readvariableop)
%savev2_opt_3_bias_read_readvariableop+
'savev2_gsc_5_kernel_read_readvariableop)
%savev2_gsc_5_bias_read_readvariableop+
'savev2_opt_5_kernel_read_readvariableop)
%savev2_opt_5_bias_read_readvariableop+
'savev2_gsc_6_kernel_read_readvariableop)
%savev2_gsc_6_bias_read_readvariableop+
'savev2_opt_6_kernel_read_readvariableop)
%savev2_opt_6_bias_read_readvariableop+
'savev2_gsc_8_kernel_read_readvariableop)
%savev2_gsc_8_bias_read_readvariableop+
'savev2_opt_8_kernel_read_readvariableop)
%savev2_opt_8_bias_read_readvariableop+
'savev2_gsc_9_kernel_read_readvariableop)
%savev2_gsc_9_bias_read_readvariableop+
'savev2_opt_9_kernel_read_readvariableop)
%savev2_opt_9_bias_read_readvariableop,
(savev2_gsc_11_kernel_read_readvariableop*
&savev2_gsc_11_bias_read_readvariableop,
(savev2_opt_11_kernel_read_readvariableop*
&savev2_opt_11_bias_read_readvariableop,
(savev2_gsc_12_kernel_read_readvariableop*
&savev2_gsc_12_bias_read_readvariableop,
(savev2_opt_12_kernel_read_readvariableop*
&savev2_opt_12_bias_read_readvariableop-
)savev2_merge_1_kernel_read_readvariableop+
'savev2_merge_1_bias_read_readvariableop-
)savev2_merge_2_kernel_read_readvariableop+
'savev2_merge_2_bias_read_readvariableop-
)savev2_merge_4_kernel_read_readvariableop+
'savev2_merge_4_bias_read_readvariableop-
)savev2_merge_5_kernel_read_readvariableop+
'savev2_merge_5_bias_read_readvariableop-
)savev2_merge_7_kernel_read_readvariableop+
'savev2_merge_7_bias_read_readvariableop-
)savev2_merge_8_kernel_read_readvariableop+
'savev2_merge_8_bias_read_readvariableop*
&savev2_fc_1_kernel_read_readvariableop(
$savev2_fc_1_bias_read_readvariableop*
&savev2_fc_3_kernel_read_readvariableop(
$savev2_fc_3_bias_read_readvariableop*
&savev2_pred_kernel_read_readvariableop(
$savev2_pred_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_gsc_2_kernel_m_read_readvariableop0
,savev2_adam_gsc_2_bias_m_read_readvariableop2
.savev2_adam_opt_2_kernel_m_read_readvariableop0
,savev2_adam_opt_2_bias_m_read_readvariableop2
.savev2_adam_gsc_3_kernel_m_read_readvariableop0
,savev2_adam_gsc_3_bias_m_read_readvariableop2
.savev2_adam_opt_3_kernel_m_read_readvariableop0
,savev2_adam_opt_3_bias_m_read_readvariableop2
.savev2_adam_gsc_5_kernel_m_read_readvariableop0
,savev2_adam_gsc_5_bias_m_read_readvariableop2
.savev2_adam_opt_5_kernel_m_read_readvariableop0
,savev2_adam_opt_5_bias_m_read_readvariableop2
.savev2_adam_gsc_6_kernel_m_read_readvariableop0
,savev2_adam_gsc_6_bias_m_read_readvariableop2
.savev2_adam_opt_6_kernel_m_read_readvariableop0
,savev2_adam_opt_6_bias_m_read_readvariableop2
.savev2_adam_gsc_8_kernel_m_read_readvariableop0
,savev2_adam_gsc_8_bias_m_read_readvariableop2
.savev2_adam_opt_8_kernel_m_read_readvariableop0
,savev2_adam_opt_8_bias_m_read_readvariableop2
.savev2_adam_gsc_9_kernel_m_read_readvariableop0
,savev2_adam_gsc_9_bias_m_read_readvariableop2
.savev2_adam_opt_9_kernel_m_read_readvariableop0
,savev2_adam_opt_9_bias_m_read_readvariableop3
/savev2_adam_gsc_11_kernel_m_read_readvariableop1
-savev2_adam_gsc_11_bias_m_read_readvariableop3
/savev2_adam_opt_11_kernel_m_read_readvariableop1
-savev2_adam_opt_11_bias_m_read_readvariableop3
/savev2_adam_gsc_12_kernel_m_read_readvariableop1
-savev2_adam_gsc_12_bias_m_read_readvariableop3
/savev2_adam_opt_12_kernel_m_read_readvariableop1
-savev2_adam_opt_12_bias_m_read_readvariableop4
0savev2_adam_merge_1_kernel_m_read_readvariableop2
.savev2_adam_merge_1_bias_m_read_readvariableop4
0savev2_adam_merge_2_kernel_m_read_readvariableop2
.savev2_adam_merge_2_bias_m_read_readvariableop4
0savev2_adam_merge_4_kernel_m_read_readvariableop2
.savev2_adam_merge_4_bias_m_read_readvariableop4
0savev2_adam_merge_5_kernel_m_read_readvariableop2
.savev2_adam_merge_5_bias_m_read_readvariableop4
0savev2_adam_merge_7_kernel_m_read_readvariableop2
.savev2_adam_merge_7_bias_m_read_readvariableop4
0savev2_adam_merge_8_kernel_m_read_readvariableop2
.savev2_adam_merge_8_bias_m_read_readvariableop1
-savev2_adam_fc_1_kernel_m_read_readvariableop/
+savev2_adam_fc_1_bias_m_read_readvariableop1
-savev2_adam_fc_3_kernel_m_read_readvariableop/
+savev2_adam_fc_3_bias_m_read_readvariableop1
-savev2_adam_pred_kernel_m_read_readvariableop/
+savev2_adam_pred_bias_m_read_readvariableop2
.savev2_adam_gsc_2_kernel_v_read_readvariableop0
,savev2_adam_gsc_2_bias_v_read_readvariableop2
.savev2_adam_opt_2_kernel_v_read_readvariableop0
,savev2_adam_opt_2_bias_v_read_readvariableop2
.savev2_adam_gsc_3_kernel_v_read_readvariableop0
,savev2_adam_gsc_3_bias_v_read_readvariableop2
.savev2_adam_opt_3_kernel_v_read_readvariableop0
,savev2_adam_opt_3_bias_v_read_readvariableop2
.savev2_adam_gsc_5_kernel_v_read_readvariableop0
,savev2_adam_gsc_5_bias_v_read_readvariableop2
.savev2_adam_opt_5_kernel_v_read_readvariableop0
,savev2_adam_opt_5_bias_v_read_readvariableop2
.savev2_adam_gsc_6_kernel_v_read_readvariableop0
,savev2_adam_gsc_6_bias_v_read_readvariableop2
.savev2_adam_opt_6_kernel_v_read_readvariableop0
,savev2_adam_opt_6_bias_v_read_readvariableop2
.savev2_adam_gsc_8_kernel_v_read_readvariableop0
,savev2_adam_gsc_8_bias_v_read_readvariableop2
.savev2_adam_opt_8_kernel_v_read_readvariableop0
,savev2_adam_opt_8_bias_v_read_readvariableop2
.savev2_adam_gsc_9_kernel_v_read_readvariableop0
,savev2_adam_gsc_9_bias_v_read_readvariableop2
.savev2_adam_opt_9_kernel_v_read_readvariableop0
,savev2_adam_opt_9_bias_v_read_readvariableop3
/savev2_adam_gsc_11_kernel_v_read_readvariableop1
-savev2_adam_gsc_11_bias_v_read_readvariableop3
/savev2_adam_opt_11_kernel_v_read_readvariableop1
-savev2_adam_opt_11_bias_v_read_readvariableop3
/savev2_adam_gsc_12_kernel_v_read_readvariableop1
-savev2_adam_gsc_12_bias_v_read_readvariableop3
/savev2_adam_opt_12_kernel_v_read_readvariableop1
-savev2_adam_opt_12_bias_v_read_readvariableop4
0savev2_adam_merge_1_kernel_v_read_readvariableop2
.savev2_adam_merge_1_bias_v_read_readvariableop4
0savev2_adam_merge_2_kernel_v_read_readvariableop2
.savev2_adam_merge_2_bias_v_read_readvariableop4
0savev2_adam_merge_4_kernel_v_read_readvariableop2
.savev2_adam_merge_4_bias_v_read_readvariableop4
0savev2_adam_merge_5_kernel_v_read_readvariableop2
.savev2_adam_merge_5_bias_v_read_readvariableop4
0savev2_adam_merge_7_kernel_v_read_readvariableop2
.savev2_adam_merge_7_bias_v_read_readvariableop4
0savev2_adam_merge_8_kernel_v_read_readvariableop2
.savev2_adam_merge_8_bias_v_read_readvariableop1
-savev2_adam_fc_1_kernel_v_read_readvariableop/
+savev2_adam_fc_1_bias_v_read_readvariableop1
-savev2_adam_fc_3_kernel_v_read_readvariableop/
+savev2_adam_fc_3_bias_v_read_readvariableop1
-savev2_adam_pred_kernel_v_read_readvariableop/
+savev2_adam_pred_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?]
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?\
value?\B?\?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?;
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_gsc_2_kernel_read_readvariableop%savev2_gsc_2_bias_read_readvariableop'savev2_opt_2_kernel_read_readvariableop%savev2_opt_2_bias_read_readvariableop'savev2_gsc_3_kernel_read_readvariableop%savev2_gsc_3_bias_read_readvariableop'savev2_opt_3_kernel_read_readvariableop%savev2_opt_3_bias_read_readvariableop'savev2_gsc_5_kernel_read_readvariableop%savev2_gsc_5_bias_read_readvariableop'savev2_opt_5_kernel_read_readvariableop%savev2_opt_5_bias_read_readvariableop'savev2_gsc_6_kernel_read_readvariableop%savev2_gsc_6_bias_read_readvariableop'savev2_opt_6_kernel_read_readvariableop%savev2_opt_6_bias_read_readvariableop'savev2_gsc_8_kernel_read_readvariableop%savev2_gsc_8_bias_read_readvariableop'savev2_opt_8_kernel_read_readvariableop%savev2_opt_8_bias_read_readvariableop'savev2_gsc_9_kernel_read_readvariableop%savev2_gsc_9_bias_read_readvariableop'savev2_opt_9_kernel_read_readvariableop%savev2_opt_9_bias_read_readvariableop(savev2_gsc_11_kernel_read_readvariableop&savev2_gsc_11_bias_read_readvariableop(savev2_opt_11_kernel_read_readvariableop&savev2_opt_11_bias_read_readvariableop(savev2_gsc_12_kernel_read_readvariableop&savev2_gsc_12_bias_read_readvariableop(savev2_opt_12_kernel_read_readvariableop&savev2_opt_12_bias_read_readvariableop)savev2_merge_1_kernel_read_readvariableop'savev2_merge_1_bias_read_readvariableop)savev2_merge_2_kernel_read_readvariableop'savev2_merge_2_bias_read_readvariableop)savev2_merge_4_kernel_read_readvariableop'savev2_merge_4_bias_read_readvariableop)savev2_merge_5_kernel_read_readvariableop'savev2_merge_5_bias_read_readvariableop)savev2_merge_7_kernel_read_readvariableop'savev2_merge_7_bias_read_readvariableop)savev2_merge_8_kernel_read_readvariableop'savev2_merge_8_bias_read_readvariableop&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop&savev2_fc_3_kernel_read_readvariableop$savev2_fc_3_bias_read_readvariableop&savev2_pred_kernel_read_readvariableop$savev2_pred_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_gsc_2_kernel_m_read_readvariableop,savev2_adam_gsc_2_bias_m_read_readvariableop.savev2_adam_opt_2_kernel_m_read_readvariableop,savev2_adam_opt_2_bias_m_read_readvariableop.savev2_adam_gsc_3_kernel_m_read_readvariableop,savev2_adam_gsc_3_bias_m_read_readvariableop.savev2_adam_opt_3_kernel_m_read_readvariableop,savev2_adam_opt_3_bias_m_read_readvariableop.savev2_adam_gsc_5_kernel_m_read_readvariableop,savev2_adam_gsc_5_bias_m_read_readvariableop.savev2_adam_opt_5_kernel_m_read_readvariableop,savev2_adam_opt_5_bias_m_read_readvariableop.savev2_adam_gsc_6_kernel_m_read_readvariableop,savev2_adam_gsc_6_bias_m_read_readvariableop.savev2_adam_opt_6_kernel_m_read_readvariableop,savev2_adam_opt_6_bias_m_read_readvariableop.savev2_adam_gsc_8_kernel_m_read_readvariableop,savev2_adam_gsc_8_bias_m_read_readvariableop.savev2_adam_opt_8_kernel_m_read_readvariableop,savev2_adam_opt_8_bias_m_read_readvariableop.savev2_adam_gsc_9_kernel_m_read_readvariableop,savev2_adam_gsc_9_bias_m_read_readvariableop.savev2_adam_opt_9_kernel_m_read_readvariableop,savev2_adam_opt_9_bias_m_read_readvariableop/savev2_adam_gsc_11_kernel_m_read_readvariableop-savev2_adam_gsc_11_bias_m_read_readvariableop/savev2_adam_opt_11_kernel_m_read_readvariableop-savev2_adam_opt_11_bias_m_read_readvariableop/savev2_adam_gsc_12_kernel_m_read_readvariableop-savev2_adam_gsc_12_bias_m_read_readvariableop/savev2_adam_opt_12_kernel_m_read_readvariableop-savev2_adam_opt_12_bias_m_read_readvariableop0savev2_adam_merge_1_kernel_m_read_readvariableop.savev2_adam_merge_1_bias_m_read_readvariableop0savev2_adam_merge_2_kernel_m_read_readvariableop.savev2_adam_merge_2_bias_m_read_readvariableop0savev2_adam_merge_4_kernel_m_read_readvariableop.savev2_adam_merge_4_bias_m_read_readvariableop0savev2_adam_merge_5_kernel_m_read_readvariableop.savev2_adam_merge_5_bias_m_read_readvariableop0savev2_adam_merge_7_kernel_m_read_readvariableop.savev2_adam_merge_7_bias_m_read_readvariableop0savev2_adam_merge_8_kernel_m_read_readvariableop.savev2_adam_merge_8_bias_m_read_readvariableop-savev2_adam_fc_1_kernel_m_read_readvariableop+savev2_adam_fc_1_bias_m_read_readvariableop-savev2_adam_fc_3_kernel_m_read_readvariableop+savev2_adam_fc_3_bias_m_read_readvariableop-savev2_adam_pred_kernel_m_read_readvariableop+savev2_adam_pred_bias_m_read_readvariableop.savev2_adam_gsc_2_kernel_v_read_readvariableop,savev2_adam_gsc_2_bias_v_read_readvariableop.savev2_adam_opt_2_kernel_v_read_readvariableop,savev2_adam_opt_2_bias_v_read_readvariableop.savev2_adam_gsc_3_kernel_v_read_readvariableop,savev2_adam_gsc_3_bias_v_read_readvariableop.savev2_adam_opt_3_kernel_v_read_readvariableop,savev2_adam_opt_3_bias_v_read_readvariableop.savev2_adam_gsc_5_kernel_v_read_readvariableop,savev2_adam_gsc_5_bias_v_read_readvariableop.savev2_adam_opt_5_kernel_v_read_readvariableop,savev2_adam_opt_5_bias_v_read_readvariableop.savev2_adam_gsc_6_kernel_v_read_readvariableop,savev2_adam_gsc_6_bias_v_read_readvariableop.savev2_adam_opt_6_kernel_v_read_readvariableop,savev2_adam_opt_6_bias_v_read_readvariableop.savev2_adam_gsc_8_kernel_v_read_readvariableop,savev2_adam_gsc_8_bias_v_read_readvariableop.savev2_adam_opt_8_kernel_v_read_readvariableop,savev2_adam_opt_8_bias_v_read_readvariableop.savev2_adam_gsc_9_kernel_v_read_readvariableop,savev2_adam_gsc_9_bias_v_read_readvariableop.savev2_adam_opt_9_kernel_v_read_readvariableop,savev2_adam_opt_9_bias_v_read_readvariableop/savev2_adam_gsc_11_kernel_v_read_readvariableop-savev2_adam_gsc_11_bias_v_read_readvariableop/savev2_adam_opt_11_kernel_v_read_readvariableop-savev2_adam_opt_11_bias_v_read_readvariableop/savev2_adam_gsc_12_kernel_v_read_readvariableop-savev2_adam_gsc_12_bias_v_read_readvariableop/savev2_adam_opt_12_kernel_v_read_readvariableop-savev2_adam_opt_12_bias_v_read_readvariableop0savev2_adam_merge_1_kernel_v_read_readvariableop.savev2_adam_merge_1_bias_v_read_readvariableop0savev2_adam_merge_2_kernel_v_read_readvariableop.savev2_adam_merge_2_bias_v_read_readvariableop0savev2_adam_merge_4_kernel_v_read_readvariableop.savev2_adam_merge_4_bias_v_read_readvariableop0savev2_adam_merge_5_kernel_v_read_readvariableop.savev2_adam_merge_5_bias_v_read_readvariableop0savev2_adam_merge_7_kernel_v_read_readvariableop.savev2_adam_merge_7_bias_v_read_readvariableop0savev2_adam_merge_8_kernel_v_read_readvariableop.savev2_adam_merge_8_bias_v_read_readvariableop-savev2_adam_fc_1_kernel_v_read_readvariableop+savev2_adam_fc_1_bias_v_read_readvariableop-savev2_adam_fc_3_kernel_v_read_readvariableop+savev2_adam_fc_3_bias_v_read_readvariableop-savev2_adam_pred_kernel_v_read_readvariableop+savev2_adam_pred_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::::::::: : : : :  : :  : :  : :  : :  : :  : : @:@:@@:@:@@:@:@@:@:@?:?:??:?:
??:?:	? : : :: : : : : : : ::::: : ::::::::::::::::: : : : :  : :  : :  : :  : :  : :  : : @:@:@@:@:@@:@:@@:@:@?:?:??:?:
??:?:	? : : :::::::::::::::::: : : : :  : :  : :  : :  : :  : :  : : @:@:@@:@:@@:@:@@:@:@?:?:??:?:
??:?:	? : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0	,
*
_output_shapes
:: 


_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0,
*
_output_shapes
:  :  

_output_shapes
: :0!,
*
_output_shapes
: @: "

_output_shapes
:@:0#,
*
_output_shapes
:@@: $

_output_shapes
:@:0%,
*
_output_shapes
:@@: &

_output_shapes
:@:0',
*
_output_shapes
:@@: (

_output_shapes
:@:1)-
+
_output_shapes
:@?:!*

_output_shapes	
:?:2+.
,
_output_shapes
:??:!,

_output_shapes	
:?:&-"
 
_output_shapes
:
??:!.

_output_shapes	
:?:%/!

_output_shapes
:	? : 0

_output_shapes
: :$1 

_output_shapes

: : 2

_output_shapes
::3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: : :

_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
::>

_output_shapes
: :?

_output_shapes
: :0@,
*
_output_shapes
:: A

_output_shapes
::0B,
*
_output_shapes
:: C

_output_shapes
::0D,
*
_output_shapes
:: E

_output_shapes
::0F,
*
_output_shapes
:: G

_output_shapes
::0H,
*
_output_shapes
:: I

_output_shapes
::0J,
*
_output_shapes
:: K

_output_shapes
::0L,
*
_output_shapes
:: M

_output_shapes
::0N,
*
_output_shapes
:: O

_output_shapes
::0P,
*
_output_shapes
: : Q

_output_shapes
: :0R,
*
_output_shapes
: : S

_output_shapes
: :0T,
*
_output_shapes
:  : U

_output_shapes
: :0V,
*
_output_shapes
:  : W

_output_shapes
: :0X,
*
_output_shapes
:  : Y

_output_shapes
: :0Z,
*
_output_shapes
:  : [

_output_shapes
: :0\,
*
_output_shapes
:  : ]

_output_shapes
: :0^,
*
_output_shapes
:  : _

_output_shapes
: :0`,
*
_output_shapes
: @: a

_output_shapes
:@:0b,
*
_output_shapes
:@@: c

_output_shapes
:@:0d,
*
_output_shapes
:@@: e

_output_shapes
:@:0f,
*
_output_shapes
:@@: g

_output_shapes
:@:1h-
+
_output_shapes
:@?:!i

_output_shapes	
:?:2j.
,
_output_shapes
:??:!k

_output_shapes	
:?:&l"
 
_output_shapes
:
??:!m

_output_shapes	
:?:%n!

_output_shapes
:	? : o

_output_shapes
: :$p 

_output_shapes

: : q

_output_shapes
::0r,
*
_output_shapes
:: s

_output_shapes
::0t,
*
_output_shapes
:: u

_output_shapes
::0v,
*
_output_shapes
:: w

_output_shapes
::0x,
*
_output_shapes
:: y

_output_shapes
::0z,
*
_output_shapes
:: {

_output_shapes
::0|,
*
_output_shapes
:: }

_output_shapes
::0~,
*
_output_shapes
:: 

_output_shapes
::1?,
*
_output_shapes
::!?

_output_shapes
::1?,
*
_output_shapes
: :!?

_output_shapes
: :1?,
*
_output_shapes
: :!?

_output_shapes
: :1?,
*
_output_shapes
:  :!?

_output_shapes
: :1?,
*
_output_shapes
:  :!?

_output_shapes
: :1?,
*
_output_shapes
:  :!?

_output_shapes
: :1?,
*
_output_shapes
:  :!?

_output_shapes
: :1?,
*
_output_shapes
:  :!?

_output_shapes
: :1?,
*
_output_shapes
:  :!?

_output_shapes
: :1?,
*
_output_shapes
: @:!?

_output_shapes
:@:1?,
*
_output_shapes
:@@:!?

_output_shapes
:@:1?,
*
_output_shapes
:@@:!?

_output_shapes
:@:1?,
*
_output_shapes
:@@:!?

_output_shapes
:@:2?-
+
_output_shapes
:@?:"?

_output_shapes	
:?:3?.
,
_output_shapes
:??:"?

_output_shapes	
:?:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:&?!

_output_shapes
:	? :!?

_output_shapes
: :%? 

_output_shapes

: :!?

_output_shapes
::?

_output_shapes
: 
?
]
A__inference_gsc_13_layer_call_and_return_conditional_losses_68802

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_gsc_3_layer_call_and_return_conditional_losses_68954

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@??o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:?????????@??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
A__inference_opt_12_layer_call_and_return_conditional_losses_69147

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ b
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ f
IdentityIdentitySigmoid:y:0^NoOp*
T0*3
_output_shapes!
:?????????@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
??
?
C__inference_cnn_base_layer_call_and_return_conditional_losses_70033

inputs)
opt_2_69892:
opt_2_69894:)
gsc_2_69897:
gsc_2_69899:)
opt_3_69902:
opt_3_69904:)
gsc_3_69907:
gsc_3_69909:)
opt_5_69914:
opt_5_69916:)
gsc_5_69919:
gsc_5_69921:)
opt_6_69924:
opt_6_69926:)
gsc_6_69929:
gsc_6_69931:)
opt_8_69936: 
opt_8_69938: )
gsc_8_69941: 
gsc_8_69943: )
opt_9_69946:  
opt_9_69948: )
gsc_9_69951:  
gsc_9_69953: *
opt_11_69958:  
opt_11_69960: *
gsc_11_69963:  
gsc_11_69965: *
opt_12_69968:  
opt_12_69970: *
gsc_12_69973:  
gsc_12_69975: +
merge_1_69982: @
merge_1_69984:@+
merge_2_69987:@@
merge_2_69989:@+
merge_4_69993:@@
merge_4_69995:@+
merge_5_69998:@@
merge_5_70000:@,
merge_7_70004:@?
merge_7_70006:	?-
merge_8_70009:??
merge_8_70011:	?

fc_1_70016:
??

fc_1_70018:	?

fc_3_70022:	? 

fc_3_70024: 

pred_70027: 

pred_70029:
identity??fc_1/StatefulPartitionedCall?fc_2/StatefulPartitionedCall?fc_3/StatefulPartitionedCall?gsc_11/StatefulPartitionedCall?gsc_12/StatefulPartitionedCall?gsc_2/StatefulPartitionedCall?gsc_3/StatefulPartitionedCall?gsc_5/StatefulPartitionedCall?gsc_6/StatefulPartitionedCall?gsc_8/StatefulPartitionedCall?gsc_9/StatefulPartitionedCall?merge_1/StatefulPartitionedCall?merge_2/StatefulPartitionedCall?merge_4/StatefulPartitionedCall?merge_5/StatefulPartitionedCall?merge_7/StatefulPartitionedCall?merge_8/StatefulPartitionedCall?opt_11/StatefulPartitionedCall?opt_12/StatefulPartitionedCall?opt_2/StatefulPartitionedCall?opt_3/StatefulPartitionedCall?opt_5/StatefulPartitionedCall?opt_6/StatefulPartitionedCall?opt_8/StatefulPartitionedCall?opt_9/StatefulPartitionedCall?pred/StatefulPartitionedCall?
opt_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_1_layer_call_and_return_conditional_losses_69777?
gsc_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_1_layer_call_and_return_conditional_losses_69758?
opt_2/StatefulPartitionedCallStatefulPartitionedCallopt_1/PartitionedCall:output:0opt_2_69892opt_2_69894*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_2_layer_call_and_return_conditional_losses_68903?
gsc_2/StatefulPartitionedCallStatefulPartitionedCallgsc_1/PartitionedCall:output:0gsc_2_69897gsc_2_69899*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_2_layer_call_and_return_conditional_losses_68920?
opt_3/StatefulPartitionedCallStatefulPartitionedCall&opt_2/StatefulPartitionedCall:output:0opt_3_69902opt_3_69904*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_3_layer_call_and_return_conditional_losses_68937?
gsc_3/StatefulPartitionedCallStatefulPartitionedCall&gsc_2/StatefulPartitionedCall:output:0gsc_3_69907gsc_3_69909*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_3_layer_call_and_return_conditional_losses_68954?
opt_4/PartitionedCallPartitionedCall&opt_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_4_layer_call_and_return_conditional_losses_68742?
gsc_4/PartitionedCallPartitionedCall&gsc_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_4_layer_call_and_return_conditional_losses_68730?
opt_5/StatefulPartitionedCallStatefulPartitionedCallopt_4/PartitionedCall:output:0opt_5_69914opt_5_69916*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_5_layer_call_and_return_conditional_losses_68973?
gsc_5/StatefulPartitionedCallStatefulPartitionedCallgsc_4/PartitionedCall:output:0gsc_5_69919gsc_5_69921*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_5_layer_call_and_return_conditional_losses_68990?
opt_6/StatefulPartitionedCallStatefulPartitionedCall&opt_5/StatefulPartitionedCall:output:0opt_6_69924opt_6_69926*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_6_layer_call_and_return_conditional_losses_69007?
gsc_6/StatefulPartitionedCallStatefulPartitionedCall&gsc_5/StatefulPartitionedCall:output:0gsc_6_69929gsc_6_69931*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_6_layer_call_and_return_conditional_losses_69024?
opt_7/PartitionedCallPartitionedCall&opt_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_7_layer_call_and_return_conditional_losses_68766?
gsc_7/PartitionedCallPartitionedCall&gsc_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_7_layer_call_and_return_conditional_losses_68754?
opt_8/StatefulPartitionedCallStatefulPartitionedCallopt_7/PartitionedCall:output:0opt_8_69936opt_8_69938*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_8_layer_call_and_return_conditional_losses_69043?
gsc_8/StatefulPartitionedCallStatefulPartitionedCallgsc_7/PartitionedCall:output:0gsc_8_69941gsc_8_69943*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_8_layer_call_and_return_conditional_losses_69060?
opt_9/StatefulPartitionedCallStatefulPartitionedCall&opt_8/StatefulPartitionedCall:output:0opt_9_69946opt_9_69948*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_9_layer_call_and_return_conditional_losses_69077?
gsc_9/StatefulPartitionedCallStatefulPartitionedCall&gsc_8/StatefulPartitionedCall:output:0gsc_9_69951gsc_9_69953*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_9_layer_call_and_return_conditional_losses_69094?
opt_10/PartitionedCallPartitionedCall&opt_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_10_layer_call_and_return_conditional_losses_68790?
gsc_10/PartitionedCallPartitionedCall&gsc_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_10_layer_call_and_return_conditional_losses_68778?
opt_11/StatefulPartitionedCallStatefulPartitionedCallopt_10/PartitionedCall:output:0opt_11_69958opt_11_69960*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_11_layer_call_and_return_conditional_losses_69113?
gsc_11/StatefulPartitionedCallStatefulPartitionedCallgsc_10/PartitionedCall:output:0gsc_11_69963gsc_11_69965*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_11_layer_call_and_return_conditional_losses_69130?
opt_12/StatefulPartitionedCallStatefulPartitionedCall'opt_11/StatefulPartitionedCall:output:0opt_12_69968opt_12_69970*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_12_layer_call_and_return_conditional_losses_69147?
gsc_12/StatefulPartitionedCallStatefulPartitionedCall'gsc_11/StatefulPartitionedCall:output:0gsc_12_69973gsc_12_69975*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_12_layer_call_and_return_conditional_losses_69164?
gsc_13/PartitionedCallPartitionedCall'gsc_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_13_layer_call_and_return_conditional_losses_68802?
opt_13/PartitionedCallPartitionedCall'opt_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_13_layer_call_and_return_conditional_losses_68814?
fuse_1/PartitionedCallPartitionedCallgsc_13/PartitionedCall:output:0opt_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_fuse_1_layer_call_and_return_conditional_losses_69178?
fuse_2/PartitionedCallPartitionedCallfuse_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_fuse_2_layer_call_and_return_conditional_losses_68826?
merge_1/StatefulPartitionedCallStatefulPartitionedCallfuse_2/PartitionedCall:output:0merge_1_69982merge_1_69984*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_1_layer_call_and_return_conditional_losses_69192?
merge_2/StatefulPartitionedCallStatefulPartitionedCall(merge_1/StatefulPartitionedCall:output:0merge_2_69987merge_2_69989*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_2_layer_call_and_return_conditional_losses_69209?
merge_3/PartitionedCallPartitionedCall(merge_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_3_layer_call_and_return_conditional_losses_68838?
merge_4/StatefulPartitionedCallStatefulPartitionedCall merge_3/PartitionedCall:output:0merge_4_69993merge_4_69995*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_4_layer_call_and_return_conditional_losses_69227?
merge_5/StatefulPartitionedCallStatefulPartitionedCall(merge_4/StatefulPartitionedCall:output:0merge_5_69998merge_5_70000*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_5_layer_call_and_return_conditional_losses_69244?
merge_6/PartitionedCallPartitionedCall(merge_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_6_layer_call_and_return_conditional_losses_68850?
merge_7/StatefulPartitionedCallStatefulPartitionedCall merge_6/PartitionedCall:output:0merge_7_70004merge_7_70006*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_7_layer_call_and_return_conditional_losses_69262?
merge_8/StatefulPartitionedCallStatefulPartitionedCall(merge_7/StatefulPartitionedCall:output:0merge_8_70009merge_8_70011*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_8_layer_call_and_return_conditional_losses_69279?
merge_9/PartitionedCallPartitionedCall(merge_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_9_layer_call_and_return_conditional_losses_68862?
flat/PartitionedCallPartitionedCall merge_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_flat_layer_call_and_return_conditional_losses_69292?
fc_1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0
fc_1_70016
fc_1_70018*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_1_layer_call_and_return_conditional_losses_69305?
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_2_layer_call_and_return_conditional_losses_69496?
fc_3/StatefulPartitionedCallStatefulPartitionedCall%fc_2/StatefulPartitionedCall:output:0
fc_3_70022
fc_3_70024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_3_layer_call_and_return_conditional_losses_69329?
pred/StatefulPartitionedCallStatefulPartitionedCall%fc_3/StatefulPartitionedCall:output:0
pred_70027
pred_70029*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_pred_layer_call_and_return_conditional_losses_69346t
IdentityIdentity%pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^fc_1/StatefulPartitionedCall^fc_2/StatefulPartitionedCall^fc_3/StatefulPartitionedCall^gsc_11/StatefulPartitionedCall^gsc_12/StatefulPartitionedCall^gsc_2/StatefulPartitionedCall^gsc_3/StatefulPartitionedCall^gsc_5/StatefulPartitionedCall^gsc_6/StatefulPartitionedCall^gsc_8/StatefulPartitionedCall^gsc_9/StatefulPartitionedCall ^merge_1/StatefulPartitionedCall ^merge_2/StatefulPartitionedCall ^merge_4/StatefulPartitionedCall ^merge_5/StatefulPartitionedCall ^merge_7/StatefulPartitionedCall ^merge_8/StatefulPartitionedCall^opt_11/StatefulPartitionedCall^opt_12/StatefulPartitionedCall^opt_2/StatefulPartitionedCall^opt_3/StatefulPartitionedCall^opt_5/StatefulPartitionedCall^opt_6/StatefulPartitionedCall^opt_8/StatefulPartitionedCall^opt_9/StatefulPartitionedCall^pred/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2@
gsc_11/StatefulPartitionedCallgsc_11/StatefulPartitionedCall2@
gsc_12/StatefulPartitionedCallgsc_12/StatefulPartitionedCall2>
gsc_2/StatefulPartitionedCallgsc_2/StatefulPartitionedCall2>
gsc_3/StatefulPartitionedCallgsc_3/StatefulPartitionedCall2>
gsc_5/StatefulPartitionedCallgsc_5/StatefulPartitionedCall2>
gsc_6/StatefulPartitionedCallgsc_6/StatefulPartitionedCall2>
gsc_8/StatefulPartitionedCallgsc_8/StatefulPartitionedCall2>
gsc_9/StatefulPartitionedCallgsc_9/StatefulPartitionedCall2B
merge_1/StatefulPartitionedCallmerge_1/StatefulPartitionedCall2B
merge_2/StatefulPartitionedCallmerge_2/StatefulPartitionedCall2B
merge_4/StatefulPartitionedCallmerge_4/StatefulPartitionedCall2B
merge_5/StatefulPartitionedCallmerge_5/StatefulPartitionedCall2B
merge_7/StatefulPartitionedCallmerge_7/StatefulPartitionedCall2B
merge_8/StatefulPartitionedCallmerge_8/StatefulPartitionedCall2@
opt_11/StatefulPartitionedCallopt_11/StatefulPartitionedCall2@
opt_12/StatefulPartitionedCallopt_12/StatefulPartitionedCall2>
opt_2/StatefulPartitionedCallopt_2/StatefulPartitionedCall2>
opt_3/StatefulPartitionedCallopt_3/StatefulPartitionedCall2>
opt_5/StatefulPartitionedCallopt_5/StatefulPartitionedCall2>
opt_6/StatefulPartitionedCallopt_6/StatefulPartitionedCall2>
opt_8/StatefulPartitionedCallopt_8/StatefulPartitionedCall2>
opt_9/StatefulPartitionedCallopt_9/StatefulPartitionedCall2<
pred/StatefulPartitionedCallpred/StatefulPartitionedCall:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
'__inference_merge_5_layer_call_fn_71822

inputs%
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_5_layer_call_and_return_conditional_losses_69244{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?	
^
?__inference_fc_2_layer_call_and_return_conditional_losses_69496

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_gsc_6_layer_call_fn_71470

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_6_layer_call_and_return_conditional_losses_69024{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@pp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
@
$__inference_fc_2_layer_call_fn_71929

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_2_layer_call_and_return_conditional_losses_69316a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
\
@__inference_gsc_4_layer_call_and_return_conditional_losses_71411

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
A
%__inference_opt_7_layer_call_fn_71516

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_7_layer_call_and_return_conditional_losses_68766?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_gsc_11_layer_call_and_return_conditional_losses_69130

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
?
?
A__inference_opt_12_layer_call_and_return_conditional_losses_71701

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ b
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ f
IdentityIdentitySigmoid:y:0^NoOp*
T0*3
_output_shapes!
:?????????@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
?
\
@__inference_gsc_1_layer_call_and_return_conditional_losses_71287

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*

begin_mask*
ellipsis_maskl
IdentityIdentitystrided_slice:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
@
$__inference_flat_layer_call_fn_71898

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_flat_layer_call_and_return_conditional_losses_69292a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
?
$__inference_fc_1_layer_call_fn_71913

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_1_layer_call_and_return_conditional_losses_69305p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
\
@__inference_opt_7_layer_call_and_return_conditional_losses_71521

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_opt_5_layer_call_and_return_conditional_losses_71461

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ppm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
?
@__inference_opt_9_layer_call_and_return_conditional_losses_71601

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 b
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 f
IdentityIdentitySigmoid:y:0^NoOp*
T0*3
_output_shapes!
:?????????@88 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@88 
 
_user_specified_nameinputs
?
?
A__inference_gsc_11_layer_call_and_return_conditional_losses_71641

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
?
\
@__inference_gsc_1_layer_call_and_return_conditional_losses_71295

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*

begin_mask*
ellipsis_maskl
IdentityIdentitystrided_slice:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
C
'__inference_merge_3_layer_call_fn_71788

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_3_layer_call_and_return_conditional_losses_68838?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_merge_3_layer_call_and_return_conditional_losses_71793

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_gsc_2_layer_call_fn_71330

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_2_layer_call_and_return_conditional_losses_68920}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:?????????@??`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
%__inference_opt_9_layer_call_fn_71590

inputs%
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_9_layer_call_and_return_conditional_losses_69077{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@88 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88 : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@88 
 
_user_specified_nameinputs
?
?
$__inference_fc_3_layer_call_fn_71960

inputs
unknown:	? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_3_layer_call_and_return_conditional_losses_69329o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_opt_6_layer_call_and_return_conditional_losses_69007

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ppm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
A
%__inference_opt_1_layer_call_fn_71300

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_1_layer_call_and_return_conditional_losses_68880n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
B__inference_merge_5_layer_call_and_return_conditional_losses_71833

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
?
@__inference_opt_2_layer_call_and_return_conditional_losses_68903

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@??o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:?????????@??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
%__inference_gsc_5_layer_call_fn_71430

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_5_layer_call_and_return_conditional_losses_68990{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@pp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
]
?__inference_fc_2_layer_call_and_return_conditional_losses_71939

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_gsc_8_layer_call_and_return_conditional_losses_69060

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@88 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@88
 
_user_specified_nameinputs
?
A
%__inference_gsc_4_layer_call_fn_71406

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_4_layer_call_and_return_conditional_losses_68730?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_opt_8_layer_call_fn_71550

inputs%
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_8_layer_call_and_return_conditional_losses_69043{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@88 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@88
 
_user_specified_nameinputs
?

?
?__inference_fc_1_layer_call_and_return_conditional_losses_69305

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_merge_9_layer_call_fn_71888

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_9_layer_call_and_return_conditional_losses_68862?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_71269
input_1%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15: 

unknown_16: (

unknown_17: 

unknown_18: (

unknown_19:  

unknown_20: (

unknown_21:  

unknown_22: (

unknown_23:  

unknown_24: (

unknown_25:  

unknown_26: (

unknown_27:  

unknown_28: (

unknown_29:  

unknown_30: (

unknown_31: @

unknown_32:@(

unknown_33:@@

unknown_34:@(

unknown_35:@@

unknown_36:@(

unknown_37:@@

unknown_38:@)

unknown_39:@?

unknown_40:	?*

unknown_41:??

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:	? 

unknown_46: 

unknown_47: 

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_68721o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:?????????@??
!
_user_specified_name	input_1
?
?
%__inference_gsc_3_layer_call_fn_71370

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_3_layer_call_and_return_conditional_losses_68954}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:?????????@??`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
]
A__inference_opt_13_layer_call_and_return_conditional_losses_71721

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
B
&__inference_opt_13_layer_call_fn_71716

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_13_layer_call_and_return_conditional_losses_68814?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
\
@__inference_gsc_4_layer_call_and_return_conditional_losses_68730

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
\
@__inference_opt_1_layer_call_and_return_conditional_losses_71321

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*
ellipsis_mask*
end_maskl
IdentityIdentitystrided_slice:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
%__inference_opt_6_layer_call_fn_71490

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_6_layer_call_and_return_conditional_losses_69007{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@pp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
?
B__inference_merge_4_layer_call_and_return_conditional_losses_71813

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_cnn_base_layer_call_fn_70644

inputs%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15: 

unknown_16: (

unknown_17: 

unknown_18: (

unknown_19:  

unknown_20: (

unknown_21:  

unknown_22: (

unknown_23:  

unknown_24: (

unknown_25:  

unknown_26: (

unknown_27:  

unknown_28: (

unknown_29:  

unknown_30: (

unknown_31: @

unknown_32:@(

unknown_33:@@

unknown_34:@(

unknown_35:@@

unknown_36:@(

unknown_37:@@

unknown_38:@)

unknown_39:@?

unknown_40:	?*

unknown_41:??

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:	? 

unknown_46: 

unknown_47: 

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_cnn_base_layer_call_and_return_conditional_losses_69353o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
??
?
C__inference_cnn_base_layer_call_and_return_conditional_losses_70533
input_1)
opt_2_70392:
opt_2_70394:)
gsc_2_70397:
gsc_2_70399:)
opt_3_70402:
opt_3_70404:)
gsc_3_70407:
gsc_3_70409:)
opt_5_70414:
opt_5_70416:)
gsc_5_70419:
gsc_5_70421:)
opt_6_70424:
opt_6_70426:)
gsc_6_70429:
gsc_6_70431:)
opt_8_70436: 
opt_8_70438: )
gsc_8_70441: 
gsc_8_70443: )
opt_9_70446:  
opt_9_70448: )
gsc_9_70451:  
gsc_9_70453: *
opt_11_70458:  
opt_11_70460: *
gsc_11_70463:  
gsc_11_70465: *
opt_12_70468:  
opt_12_70470: *
gsc_12_70473:  
gsc_12_70475: +
merge_1_70482: @
merge_1_70484:@+
merge_2_70487:@@
merge_2_70489:@+
merge_4_70493:@@
merge_4_70495:@+
merge_5_70498:@@
merge_5_70500:@,
merge_7_70504:@?
merge_7_70506:	?-
merge_8_70509:??
merge_8_70511:	?

fc_1_70516:
??

fc_1_70518:	?

fc_3_70522:	? 

fc_3_70524: 

pred_70527: 

pred_70529:
identity??fc_1/StatefulPartitionedCall?fc_2/StatefulPartitionedCall?fc_3/StatefulPartitionedCall?gsc_11/StatefulPartitionedCall?gsc_12/StatefulPartitionedCall?gsc_2/StatefulPartitionedCall?gsc_3/StatefulPartitionedCall?gsc_5/StatefulPartitionedCall?gsc_6/StatefulPartitionedCall?gsc_8/StatefulPartitionedCall?gsc_9/StatefulPartitionedCall?merge_1/StatefulPartitionedCall?merge_2/StatefulPartitionedCall?merge_4/StatefulPartitionedCall?merge_5/StatefulPartitionedCall?merge_7/StatefulPartitionedCall?merge_8/StatefulPartitionedCall?opt_11/StatefulPartitionedCall?opt_12/StatefulPartitionedCall?opt_2/StatefulPartitionedCall?opt_3/StatefulPartitionedCall?opt_5/StatefulPartitionedCall?opt_6/StatefulPartitionedCall?opt_8/StatefulPartitionedCall?opt_9/StatefulPartitionedCall?pred/StatefulPartitionedCall?
opt_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_1_layer_call_and_return_conditional_losses_69777?
gsc_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_1_layer_call_and_return_conditional_losses_69758?
opt_2/StatefulPartitionedCallStatefulPartitionedCallopt_1/PartitionedCall:output:0opt_2_70392opt_2_70394*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_2_layer_call_and_return_conditional_losses_68903?
gsc_2/StatefulPartitionedCallStatefulPartitionedCallgsc_1/PartitionedCall:output:0gsc_2_70397gsc_2_70399*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_2_layer_call_and_return_conditional_losses_68920?
opt_3/StatefulPartitionedCallStatefulPartitionedCall&opt_2/StatefulPartitionedCall:output:0opt_3_70402opt_3_70404*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_3_layer_call_and_return_conditional_losses_68937?
gsc_3/StatefulPartitionedCallStatefulPartitionedCall&gsc_2/StatefulPartitionedCall:output:0gsc_3_70407gsc_3_70409*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_3_layer_call_and_return_conditional_losses_68954?
opt_4/PartitionedCallPartitionedCall&opt_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_4_layer_call_and_return_conditional_losses_68742?
gsc_4/PartitionedCallPartitionedCall&gsc_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_4_layer_call_and_return_conditional_losses_68730?
opt_5/StatefulPartitionedCallStatefulPartitionedCallopt_4/PartitionedCall:output:0opt_5_70414opt_5_70416*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_5_layer_call_and_return_conditional_losses_68973?
gsc_5/StatefulPartitionedCallStatefulPartitionedCallgsc_4/PartitionedCall:output:0gsc_5_70419gsc_5_70421*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_5_layer_call_and_return_conditional_losses_68990?
opt_6/StatefulPartitionedCallStatefulPartitionedCall&opt_5/StatefulPartitionedCall:output:0opt_6_70424opt_6_70426*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_6_layer_call_and_return_conditional_losses_69007?
gsc_6/StatefulPartitionedCallStatefulPartitionedCall&gsc_5/StatefulPartitionedCall:output:0gsc_6_70429gsc_6_70431*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_6_layer_call_and_return_conditional_losses_69024?
opt_7/PartitionedCallPartitionedCall&opt_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_7_layer_call_and_return_conditional_losses_68766?
gsc_7/PartitionedCallPartitionedCall&gsc_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_7_layer_call_and_return_conditional_losses_68754?
opt_8/StatefulPartitionedCallStatefulPartitionedCallopt_7/PartitionedCall:output:0opt_8_70436opt_8_70438*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_8_layer_call_and_return_conditional_losses_69043?
gsc_8/StatefulPartitionedCallStatefulPartitionedCallgsc_7/PartitionedCall:output:0gsc_8_70441gsc_8_70443*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_8_layer_call_and_return_conditional_losses_69060?
opt_9/StatefulPartitionedCallStatefulPartitionedCall&opt_8/StatefulPartitionedCall:output:0opt_9_70446opt_9_70448*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_9_layer_call_and_return_conditional_losses_69077?
gsc_9/StatefulPartitionedCallStatefulPartitionedCall&gsc_8/StatefulPartitionedCall:output:0gsc_9_70451gsc_9_70453*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_9_layer_call_and_return_conditional_losses_69094?
opt_10/PartitionedCallPartitionedCall&opt_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_10_layer_call_and_return_conditional_losses_68790?
gsc_10/PartitionedCallPartitionedCall&gsc_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_10_layer_call_and_return_conditional_losses_68778?
opt_11/StatefulPartitionedCallStatefulPartitionedCallopt_10/PartitionedCall:output:0opt_11_70458opt_11_70460*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_11_layer_call_and_return_conditional_losses_69113?
gsc_11/StatefulPartitionedCallStatefulPartitionedCallgsc_10/PartitionedCall:output:0gsc_11_70463gsc_11_70465*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_11_layer_call_and_return_conditional_losses_69130?
opt_12/StatefulPartitionedCallStatefulPartitionedCall'opt_11/StatefulPartitionedCall:output:0opt_12_70468opt_12_70470*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_12_layer_call_and_return_conditional_losses_69147?
gsc_12/StatefulPartitionedCallStatefulPartitionedCall'gsc_11/StatefulPartitionedCall:output:0gsc_12_70473gsc_12_70475*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_12_layer_call_and_return_conditional_losses_69164?
gsc_13/PartitionedCallPartitionedCall'gsc_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_13_layer_call_and_return_conditional_losses_68802?
opt_13/PartitionedCallPartitionedCall'opt_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_13_layer_call_and_return_conditional_losses_68814?
fuse_1/PartitionedCallPartitionedCallgsc_13/PartitionedCall:output:0opt_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_fuse_1_layer_call_and_return_conditional_losses_69178?
fuse_2/PartitionedCallPartitionedCallfuse_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_fuse_2_layer_call_and_return_conditional_losses_68826?
merge_1/StatefulPartitionedCallStatefulPartitionedCallfuse_2/PartitionedCall:output:0merge_1_70482merge_1_70484*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_1_layer_call_and_return_conditional_losses_69192?
merge_2/StatefulPartitionedCallStatefulPartitionedCall(merge_1/StatefulPartitionedCall:output:0merge_2_70487merge_2_70489*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_2_layer_call_and_return_conditional_losses_69209?
merge_3/PartitionedCallPartitionedCall(merge_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_3_layer_call_and_return_conditional_losses_68838?
merge_4/StatefulPartitionedCallStatefulPartitionedCall merge_3/PartitionedCall:output:0merge_4_70493merge_4_70495*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_4_layer_call_and_return_conditional_losses_69227?
merge_5/StatefulPartitionedCallStatefulPartitionedCall(merge_4/StatefulPartitionedCall:output:0merge_5_70498merge_5_70500*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_5_layer_call_and_return_conditional_losses_69244?
merge_6/PartitionedCallPartitionedCall(merge_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_6_layer_call_and_return_conditional_losses_68850?
merge_7/StatefulPartitionedCallStatefulPartitionedCall merge_6/PartitionedCall:output:0merge_7_70504merge_7_70506*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_7_layer_call_and_return_conditional_losses_69262?
merge_8/StatefulPartitionedCallStatefulPartitionedCall(merge_7/StatefulPartitionedCall:output:0merge_8_70509merge_8_70511*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_8_layer_call_and_return_conditional_losses_69279?
merge_9/PartitionedCallPartitionedCall(merge_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_9_layer_call_and_return_conditional_losses_68862?
flat/PartitionedCallPartitionedCall merge_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_flat_layer_call_and_return_conditional_losses_69292?
fc_1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0
fc_1_70516
fc_1_70518*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_1_layer_call_and_return_conditional_losses_69305?
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_2_layer_call_and_return_conditional_losses_69496?
fc_3/StatefulPartitionedCallStatefulPartitionedCall%fc_2/StatefulPartitionedCall:output:0
fc_3_70522
fc_3_70524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_3_layer_call_and_return_conditional_losses_69329?
pred/StatefulPartitionedCallStatefulPartitionedCall%fc_3/StatefulPartitionedCall:output:0
pred_70527
pred_70529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_pred_layer_call_and_return_conditional_losses_69346t
IdentityIdentity%pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^fc_1/StatefulPartitionedCall^fc_2/StatefulPartitionedCall^fc_3/StatefulPartitionedCall^gsc_11/StatefulPartitionedCall^gsc_12/StatefulPartitionedCall^gsc_2/StatefulPartitionedCall^gsc_3/StatefulPartitionedCall^gsc_5/StatefulPartitionedCall^gsc_6/StatefulPartitionedCall^gsc_8/StatefulPartitionedCall^gsc_9/StatefulPartitionedCall ^merge_1/StatefulPartitionedCall ^merge_2/StatefulPartitionedCall ^merge_4/StatefulPartitionedCall ^merge_5/StatefulPartitionedCall ^merge_7/StatefulPartitionedCall ^merge_8/StatefulPartitionedCall^opt_11/StatefulPartitionedCall^opt_12/StatefulPartitionedCall^opt_2/StatefulPartitionedCall^opt_3/StatefulPartitionedCall^opt_5/StatefulPartitionedCall^opt_6/StatefulPartitionedCall^opt_8/StatefulPartitionedCall^opt_9/StatefulPartitionedCall^pred/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2@
gsc_11/StatefulPartitionedCallgsc_11/StatefulPartitionedCall2@
gsc_12/StatefulPartitionedCallgsc_12/StatefulPartitionedCall2>
gsc_2/StatefulPartitionedCallgsc_2/StatefulPartitionedCall2>
gsc_3/StatefulPartitionedCallgsc_3/StatefulPartitionedCall2>
gsc_5/StatefulPartitionedCallgsc_5/StatefulPartitionedCall2>
gsc_6/StatefulPartitionedCallgsc_6/StatefulPartitionedCall2>
gsc_8/StatefulPartitionedCallgsc_8/StatefulPartitionedCall2>
gsc_9/StatefulPartitionedCallgsc_9/StatefulPartitionedCall2B
merge_1/StatefulPartitionedCallmerge_1/StatefulPartitionedCall2B
merge_2/StatefulPartitionedCallmerge_2/StatefulPartitionedCall2B
merge_4/StatefulPartitionedCallmerge_4/StatefulPartitionedCall2B
merge_5/StatefulPartitionedCallmerge_5/StatefulPartitionedCall2B
merge_7/StatefulPartitionedCallmerge_7/StatefulPartitionedCall2B
merge_8/StatefulPartitionedCallmerge_8/StatefulPartitionedCall2@
opt_11/StatefulPartitionedCallopt_11/StatefulPartitionedCall2@
opt_12/StatefulPartitionedCallopt_12/StatefulPartitionedCall2>
opt_2/StatefulPartitionedCallopt_2/StatefulPartitionedCall2>
opt_3/StatefulPartitionedCallopt_3/StatefulPartitionedCall2>
opt_5/StatefulPartitionedCallopt_5/StatefulPartitionedCall2>
opt_6/StatefulPartitionedCallopt_6/StatefulPartitionedCall2>
opt_8/StatefulPartitionedCallopt_8/StatefulPartitionedCall2>
opt_9/StatefulPartitionedCallopt_9/StatefulPartitionedCall2<
pred/StatefulPartitionedCallpred/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:?????????@??
!
_user_specified_name	input_1
?
?
B__inference_merge_2_layer_call_and_return_conditional_losses_71783

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
A
%__inference_gsc_1_layer_call_fn_71279

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_1_layer_call_and_return_conditional_losses_69758n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
%__inference_opt_5_layer_call_fn_71450

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_5_layer_call_and_return_conditional_losses_68973{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@pp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
]
A__inference_opt_13_layer_call_and_return_conditional_losses_68814

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
A
%__inference_opt_1_layer_call_fn_71305

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_1_layer_call_and_return_conditional_losses_69777n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
%__inference_opt_3_layer_call_fn_71390

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_3_layer_call_and_return_conditional_losses_68937}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:?????????@??`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
@__inference_opt_3_layer_call_and_return_conditional_losses_68937

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@??o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:?????????@??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
(__inference_cnn_base_layer_call_fn_70749

inputs%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15: 

unknown_16: (

unknown_17: 

unknown_18: (

unknown_19:  

unknown_20: (

unknown_21:  

unknown_22: (

unknown_23:  

unknown_24: (

unknown_25:  

unknown_26: (

unknown_27:  

unknown_28: (

unknown_29:  

unknown_30: (

unknown_31: @

unknown_32:@(

unknown_33:@@

unknown_34:@(

unknown_35:@@

unknown_36:@(

unknown_37:@@

unknown_38:@)

unknown_39:@?

unknown_40:	?*

unknown_41:??

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:	? 

unknown_46: 

unknown_47: 

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_cnn_base_layer_call_and_return_conditional_losses_70033o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
$__inference_pred_layer_call_fn_71980

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_pred_layer_call_and_return_conditional_losses_69346o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
@__inference_gsc_3_layer_call_and_return_conditional_losses_71381

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@??o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:?????????@??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
%__inference_gsc_9_layer_call_fn_71570

inputs%
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_9_layer_call_and_return_conditional_losses_69094{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@88 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88 : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@88 
 
_user_specified_nameinputs
?
B
&__inference_gsc_13_layer_call_fn_71706

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_13_layer_call_and_return_conditional_losses_68802?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?

?
?__inference_pred_layer_call_and_return_conditional_losses_71991

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
A
%__inference_opt_4_layer_call_fn_71416

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_4_layer_call_and_return_conditional_losses_68742?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_opt_2_layer_call_fn_71350

inputs%
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_2_layer_call_and_return_conditional_losses_68903}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:?????????@??`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
??
?$
C__inference_cnn_base_layer_call_and_return_conditional_losses_71162

inputsB
$opt_2_conv3d_readvariableop_resource:3
%opt_2_biasadd_readvariableop_resource:B
$gsc_2_conv3d_readvariableop_resource:3
%gsc_2_biasadd_readvariableop_resource:B
$opt_3_conv3d_readvariableop_resource:3
%opt_3_biasadd_readvariableop_resource:B
$gsc_3_conv3d_readvariableop_resource:3
%gsc_3_biasadd_readvariableop_resource:B
$opt_5_conv3d_readvariableop_resource:3
%opt_5_biasadd_readvariableop_resource:B
$gsc_5_conv3d_readvariableop_resource:3
%gsc_5_biasadd_readvariableop_resource:B
$opt_6_conv3d_readvariableop_resource:3
%opt_6_biasadd_readvariableop_resource:B
$gsc_6_conv3d_readvariableop_resource:3
%gsc_6_biasadd_readvariableop_resource:B
$opt_8_conv3d_readvariableop_resource: 3
%opt_8_biasadd_readvariableop_resource: B
$gsc_8_conv3d_readvariableop_resource: 3
%gsc_8_biasadd_readvariableop_resource: B
$opt_9_conv3d_readvariableop_resource:  3
%opt_9_biasadd_readvariableop_resource: B
$gsc_9_conv3d_readvariableop_resource:  3
%gsc_9_biasadd_readvariableop_resource: C
%opt_11_conv3d_readvariableop_resource:  4
&opt_11_biasadd_readvariableop_resource: C
%gsc_11_conv3d_readvariableop_resource:  4
&gsc_11_biasadd_readvariableop_resource: C
%opt_12_conv3d_readvariableop_resource:  4
&opt_12_biasadd_readvariableop_resource: C
%gsc_12_conv3d_readvariableop_resource:  4
&gsc_12_biasadd_readvariableop_resource: D
&merge_1_conv3d_readvariableop_resource: @5
'merge_1_biasadd_readvariableop_resource:@D
&merge_2_conv3d_readvariableop_resource:@@5
'merge_2_biasadd_readvariableop_resource:@D
&merge_4_conv3d_readvariableop_resource:@@5
'merge_4_biasadd_readvariableop_resource:@D
&merge_5_conv3d_readvariableop_resource:@@5
'merge_5_biasadd_readvariableop_resource:@E
&merge_7_conv3d_readvariableop_resource:@?6
'merge_7_biasadd_readvariableop_resource:	?F
&merge_8_conv3d_readvariableop_resource:??6
'merge_8_biasadd_readvariableop_resource:	?7
#fc_1_matmul_readvariableop_resource:
??3
$fc_1_biasadd_readvariableop_resource:	?6
#fc_3_matmul_readvariableop_resource:	? 2
$fc_3_biasadd_readvariableop_resource: 5
#pred_matmul_readvariableop_resource: 2
$pred_biasadd_readvariableop_resource:
identity??fc_1/BiasAdd/ReadVariableOp?fc_1/MatMul/ReadVariableOp?fc_3/BiasAdd/ReadVariableOp?fc_3/MatMul/ReadVariableOp?gsc_11/BiasAdd/ReadVariableOp?gsc_11/Conv3D/ReadVariableOp?gsc_12/BiasAdd/ReadVariableOp?gsc_12/Conv3D/ReadVariableOp?gsc_2/BiasAdd/ReadVariableOp?gsc_2/Conv3D/ReadVariableOp?gsc_3/BiasAdd/ReadVariableOp?gsc_3/Conv3D/ReadVariableOp?gsc_5/BiasAdd/ReadVariableOp?gsc_5/Conv3D/ReadVariableOp?gsc_6/BiasAdd/ReadVariableOp?gsc_6/Conv3D/ReadVariableOp?gsc_8/BiasAdd/ReadVariableOp?gsc_8/Conv3D/ReadVariableOp?gsc_9/BiasAdd/ReadVariableOp?gsc_9/Conv3D/ReadVariableOp?merge_1/BiasAdd/ReadVariableOp?merge_1/Conv3D/ReadVariableOp?merge_2/BiasAdd/ReadVariableOp?merge_2/Conv3D/ReadVariableOp?merge_4/BiasAdd/ReadVariableOp?merge_4/Conv3D/ReadVariableOp?merge_5/BiasAdd/ReadVariableOp?merge_5/Conv3D/ReadVariableOp?merge_7/BiasAdd/ReadVariableOp?merge_7/Conv3D/ReadVariableOp?merge_8/BiasAdd/ReadVariableOp?merge_8/Conv3D/ReadVariableOp?opt_11/BiasAdd/ReadVariableOp?opt_11/Conv3D/ReadVariableOp?opt_12/BiasAdd/ReadVariableOp?opt_12/Conv3D/ReadVariableOp?opt_2/BiasAdd/ReadVariableOp?opt_2/Conv3D/ReadVariableOp?opt_3/BiasAdd/ReadVariableOp?opt_3/Conv3D/ReadVariableOp?opt_5/BiasAdd/ReadVariableOp?opt_5/Conv3D/ReadVariableOp?opt_6/BiasAdd/ReadVariableOp?opt_6/Conv3D/ReadVariableOp?opt_8/BiasAdd/ReadVariableOp?opt_8/Conv3D/ReadVariableOp?opt_9/BiasAdd/ReadVariableOp?opt_9/Conv3D/ReadVariableOp?pred/BiasAdd/ReadVariableOp?pred/MatMul/ReadVariableOpj
opt_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       l
opt_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        l
opt_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
opt_1/strided_sliceStridedSliceinputs"opt_1/strided_slice/stack:output:0$opt_1/strided_slice/stack_1:output:0$opt_1/strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*
ellipsis_mask*
end_maskj
gsc_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        l
gsc_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       l
gsc_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
gsc_1/strided_sliceStridedSliceinputs"gsc_1/strided_slice/stack:output:0$gsc_1/strided_slice/stack_1:output:0$gsc_1/strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*

begin_mask*
ellipsis_mask?
opt_2/Conv3D/ReadVariableOpReadVariableOp$opt_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
opt_2/Conv3DConv3Dopt_1/strided_slice:output:0#opt_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
~
opt_2/BiasAdd/ReadVariableOpReadVariableOp%opt_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
opt_2/BiasAddBiasAddopt_2/Conv3D:output:0$opt_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??j

opt_2/ReluReluopt_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
gsc_2/Conv3D/ReadVariableOpReadVariableOp$gsc_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
gsc_2/Conv3DConv3Dgsc_1/strided_slice:output:0#gsc_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
~
gsc_2/BiasAdd/ReadVariableOpReadVariableOp%gsc_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
gsc_2/BiasAddBiasAddgsc_2/Conv3D:output:0$gsc_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??j

gsc_2/ReluRelugsc_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
opt_3/Conv3D/ReadVariableOpReadVariableOp$opt_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
opt_3/Conv3DConv3Dopt_2/Relu:activations:0#opt_3/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
~
opt_3/BiasAdd/ReadVariableOpReadVariableOp%opt_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
opt_3/BiasAddBiasAddopt_3/Conv3D:output:0$opt_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??j

opt_3/ReluReluopt_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
gsc_3/Conv3D/ReadVariableOpReadVariableOp$gsc_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
gsc_3/Conv3DConv3Dgsc_2/Relu:activations:0#gsc_3/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
~
gsc_3/BiasAdd/ReadVariableOpReadVariableOp%gsc_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
gsc_3/BiasAddBiasAddgsc_3/Conv3D:output:0$gsc_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??j

gsc_3/ReluRelugsc_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
opt_4/MaxPool3D	MaxPool3Dopt_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????@pp*
ksize	
*
paddingVALID*
strides	
?
gsc_4/MaxPool3D	MaxPool3Dgsc_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????@pp*
ksize	
*
paddingVALID*
strides	
?
opt_5/Conv3D/ReadVariableOpReadVariableOp$opt_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
opt_5/Conv3DConv3Dopt_4/MaxPool3D:output:0#opt_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
~
opt_5/BiasAdd/ReadVariableOpReadVariableOp%opt_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
opt_5/BiasAddBiasAddopt_5/Conv3D:output:0$opt_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pph

opt_5/ReluReluopt_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
gsc_5/Conv3D/ReadVariableOpReadVariableOp$gsc_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
gsc_5/Conv3DConv3Dgsc_4/MaxPool3D:output:0#gsc_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
~
gsc_5/BiasAdd/ReadVariableOpReadVariableOp%gsc_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
gsc_5/BiasAddBiasAddgsc_5/Conv3D:output:0$gsc_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pph

gsc_5/ReluRelugsc_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
opt_6/Conv3D/ReadVariableOpReadVariableOp$opt_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
opt_6/Conv3DConv3Dopt_5/Relu:activations:0#opt_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
~
opt_6/BiasAdd/ReadVariableOpReadVariableOp%opt_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
opt_6/BiasAddBiasAddopt_6/Conv3D:output:0$opt_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pph

opt_6/ReluReluopt_6/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
gsc_6/Conv3D/ReadVariableOpReadVariableOp$gsc_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
gsc_6/Conv3DConv3Dgsc_5/Relu:activations:0#gsc_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
~
gsc_6/BiasAdd/ReadVariableOpReadVariableOp%gsc_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
gsc_6/BiasAddBiasAddgsc_6/Conv3D:output:0$gsc_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pph

gsc_6/ReluRelugsc_6/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
opt_7/MaxPool3D	MaxPool3Dopt_6/Relu:activations:0*
T0*3
_output_shapes!
:?????????@88*
ksize	
*
paddingVALID*
strides	
?
gsc_7/MaxPool3D	MaxPool3Dgsc_6/Relu:activations:0*
T0*3
_output_shapes!
:?????????@88*
ksize	
*
paddingVALID*
strides	
?
opt_8/Conv3D/ReadVariableOpReadVariableOp$opt_8_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0?
opt_8/Conv3DConv3Dopt_7/MaxPool3D:output:0#opt_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
~
opt_8/BiasAdd/ReadVariableOpReadVariableOp%opt_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
opt_8/BiasAddBiasAddopt_8/Conv3D:output:0$opt_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 n
opt_8/SigmoidSigmoidopt_8/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
gsc_8/Conv3D/ReadVariableOpReadVariableOp$gsc_8_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0?
gsc_8/Conv3DConv3Dgsc_7/MaxPool3D:output:0#gsc_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
~
gsc_8/BiasAdd/ReadVariableOpReadVariableOp%gsc_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
gsc_8/BiasAddBiasAddgsc_8/Conv3D:output:0$gsc_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 h

gsc_8/ReluRelugsc_8/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
opt_9/Conv3D/ReadVariableOpReadVariableOp$opt_9_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
opt_9/Conv3DConv3Dopt_8/Sigmoid:y:0#opt_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
~
opt_9/BiasAdd/ReadVariableOpReadVariableOp%opt_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
opt_9/BiasAddBiasAddopt_9/Conv3D:output:0$opt_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 n
opt_9/SigmoidSigmoidopt_9/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
gsc_9/Conv3D/ReadVariableOpReadVariableOp$gsc_9_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
gsc_9/Conv3DConv3Dgsc_8/Relu:activations:0#gsc_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
~
gsc_9/BiasAdd/ReadVariableOpReadVariableOp%gsc_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
gsc_9/BiasAddBiasAddgsc_9/Conv3D:output:0$gsc_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 h

gsc_9/ReluRelugsc_9/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
opt_10/MaxPool3D	MaxPool3Dopt_9/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?
gsc_10/MaxPool3D	MaxPool3Dgsc_9/Relu:activations:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?
opt_11/Conv3D/ReadVariableOpReadVariableOp%opt_11_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
opt_11/Conv3DConv3Dopt_10/MaxPool3D:output:0$opt_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
opt_11/BiasAdd/ReadVariableOpReadVariableOp&opt_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
opt_11/BiasAddBiasAddopt_11/Conv3D:output:0%opt_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ p
opt_11/SigmoidSigmoidopt_11/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
gsc_11/Conv3D/ReadVariableOpReadVariableOp%gsc_11_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
gsc_11/Conv3DConv3Dgsc_10/MaxPool3D:output:0$gsc_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
gsc_11/BiasAdd/ReadVariableOpReadVariableOp&gsc_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
gsc_11/BiasAddBiasAddgsc_11/Conv3D:output:0%gsc_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ j
gsc_11/ReluRelugsc_11/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
opt_12/Conv3D/ReadVariableOpReadVariableOp%opt_12_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
opt_12/Conv3DConv3Dopt_11/Sigmoid:y:0$opt_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
opt_12/BiasAdd/ReadVariableOpReadVariableOp&opt_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
opt_12/BiasAddBiasAddopt_12/Conv3D:output:0%opt_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ p
opt_12/SigmoidSigmoidopt_12/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
gsc_12/Conv3D/ReadVariableOpReadVariableOp%gsc_12_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
gsc_12/Conv3DConv3Dgsc_11/Relu:activations:0$gsc_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
gsc_12/BiasAdd/ReadVariableOpReadVariableOp&gsc_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
gsc_12/BiasAddBiasAddgsc_12/Conv3D:output:0%gsc_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ j
gsc_12/ReluRelugsc_12/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
gsc_13/MaxPool3D	MaxPool3Dgsc_12/Relu:activations:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?
opt_13/MaxPool3D	MaxPool3Dopt_12/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?

fuse_1/mulMulgsc_13/MaxPool3D:output:0opt_13/MaxPool3D:output:0*
T0*3
_output_shapes!
:?????????@ ?
fuse_2/MaxPool3D	MaxPool3Dfuse_1/mul:z:0*
T0*3
_output_shapes!
:????????? *
ksize	
*
paddingVALID*
strides	
?
merge_1/Conv3D/ReadVariableOpReadVariableOp&merge_1_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0?
merge_1/Conv3DConv3Dfuse_2/MaxPool3D:output:0%merge_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
merge_1/BiasAdd/ReadVariableOpReadVariableOp'merge_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
merge_1/BiasAddBiasAddmerge_1/Conv3D:output:0&merge_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@l
merge_1/ReluRelumerge_1/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
merge_2/Conv3D/ReadVariableOpReadVariableOp&merge_2_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
merge_2/Conv3DConv3Dmerge_1/Relu:activations:0%merge_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
merge_2/BiasAdd/ReadVariableOpReadVariableOp'merge_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
merge_2/BiasAddBiasAddmerge_2/Conv3D:output:0&merge_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@l
merge_2/ReluRelumerge_2/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
merge_3/MaxPool3D	MaxPool3Dmerge_2/Relu:activations:0*
T0*3
_output_shapes!
:?????????@*
ksize	
*
paddingVALID*
strides	
?
merge_4/Conv3D/ReadVariableOpReadVariableOp&merge_4_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
merge_4/Conv3DConv3Dmerge_3/MaxPool3D:output:0%merge_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
merge_4/BiasAdd/ReadVariableOpReadVariableOp'merge_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
merge_4/BiasAddBiasAddmerge_4/Conv3D:output:0&merge_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@l
merge_4/ReluRelumerge_4/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
merge_5/Conv3D/ReadVariableOpReadVariableOp&merge_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
merge_5/Conv3DConv3Dmerge_4/Relu:activations:0%merge_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
merge_5/BiasAdd/ReadVariableOpReadVariableOp'merge_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
merge_5/BiasAddBiasAddmerge_5/Conv3D:output:0&merge_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@l
merge_5/ReluRelumerge_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
merge_6/MaxPool3D	MaxPool3Dmerge_5/Relu:activations:0*
T0*3
_output_shapes!
:?????????@*
ksize	
*
paddingVALID*
strides	
?
merge_7/Conv3D/ReadVariableOpReadVariableOp&merge_7_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype0?
merge_7/Conv3DConv3Dmerge_6/MaxPool3D:output:0%merge_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
?
merge_7/BiasAdd/ReadVariableOpReadVariableOp'merge_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
merge_7/BiasAddBiasAddmerge_7/Conv3D:output:0&merge_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????m
merge_7/ReluRelumerge_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :???????????
merge_8/Conv3D/ReadVariableOpReadVariableOp&merge_8_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype0?
merge_8/Conv3DConv3Dmerge_7/Relu:activations:0%merge_8/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
?
merge_8/BiasAdd/ReadVariableOpReadVariableOp'merge_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
merge_8/BiasAddBiasAddmerge_8/Conv3D:output:0&merge_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????m
merge_8/ReluRelumerge_8/BiasAdd:output:0*
T0*4
_output_shapes"
 :???????????
merge_9/MaxPool3D	MaxPool3Dmerge_8/Relu:activations:0*
T0*4
_output_shapes"
 :??????????*
ksize	
*
paddingVALID*
strides	
[

flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   {
flat/ReshapeReshapemerge_9/MaxPool3D:output:0flat/Const:output:0*
T0*(
_output_shapes
:???????????
fc_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
fc_1/MatMulMatMulflat/Reshape:output:0"fc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????}
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????W
fc_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
fc_2/dropout/MulMulfc_1/Relu:activations:0fc_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????Y
fc_2/dropout/ShapeShapefc_1/Relu:activations:0*
T0*
_output_shapes
:?
)fc_2/dropout/random_uniform/RandomUniformRandomUniformfc_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0`
fc_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
fc_2/dropout/GreaterEqualGreaterEqual2fc_2/dropout/random_uniform/RandomUniform:output:0$fc_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????z
fc_2/dropout/CastCastfc_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????y
fc_2/dropout/Mul_1Mulfc_2/dropout/Mul:z:0fc_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????
fc_3/MatMul/ReadVariableOpReadVariableOp#fc_3_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
fc_3/MatMulMatMulfc_2/dropout/Mul_1:z:0"fc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? |
fc_3/BiasAdd/ReadVariableOpReadVariableOp$fc_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
fc_3/BiasAddBiasAddfc_3/MatMul:product:0#fc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? Z
	fc_3/ReluRelufc_3/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ~
pred/MatMul/ReadVariableOpReadVariableOp#pred_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
pred/MatMulMatMulfc_3/Relu:activations:0"pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
pred/BiasAdd/ReadVariableOpReadVariableOp$pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
pred/BiasAddBiasAddpred/MatMul:product:0#pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
pred/SigmoidSigmoidpred/BiasAdd:output:0*
T0*'
_output_shapes
:?????????_
IdentityIdentitypred/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^fc_1/BiasAdd/ReadVariableOp^fc_1/MatMul/ReadVariableOp^fc_3/BiasAdd/ReadVariableOp^fc_3/MatMul/ReadVariableOp^gsc_11/BiasAdd/ReadVariableOp^gsc_11/Conv3D/ReadVariableOp^gsc_12/BiasAdd/ReadVariableOp^gsc_12/Conv3D/ReadVariableOp^gsc_2/BiasAdd/ReadVariableOp^gsc_2/Conv3D/ReadVariableOp^gsc_3/BiasAdd/ReadVariableOp^gsc_3/Conv3D/ReadVariableOp^gsc_5/BiasAdd/ReadVariableOp^gsc_5/Conv3D/ReadVariableOp^gsc_6/BiasAdd/ReadVariableOp^gsc_6/Conv3D/ReadVariableOp^gsc_8/BiasAdd/ReadVariableOp^gsc_8/Conv3D/ReadVariableOp^gsc_9/BiasAdd/ReadVariableOp^gsc_9/Conv3D/ReadVariableOp^merge_1/BiasAdd/ReadVariableOp^merge_1/Conv3D/ReadVariableOp^merge_2/BiasAdd/ReadVariableOp^merge_2/Conv3D/ReadVariableOp^merge_4/BiasAdd/ReadVariableOp^merge_4/Conv3D/ReadVariableOp^merge_5/BiasAdd/ReadVariableOp^merge_5/Conv3D/ReadVariableOp^merge_7/BiasAdd/ReadVariableOp^merge_7/Conv3D/ReadVariableOp^merge_8/BiasAdd/ReadVariableOp^merge_8/Conv3D/ReadVariableOp^opt_11/BiasAdd/ReadVariableOp^opt_11/Conv3D/ReadVariableOp^opt_12/BiasAdd/ReadVariableOp^opt_12/Conv3D/ReadVariableOp^opt_2/BiasAdd/ReadVariableOp^opt_2/Conv3D/ReadVariableOp^opt_3/BiasAdd/ReadVariableOp^opt_3/Conv3D/ReadVariableOp^opt_5/BiasAdd/ReadVariableOp^opt_5/Conv3D/ReadVariableOp^opt_6/BiasAdd/ReadVariableOp^opt_6/Conv3D/ReadVariableOp^opt_8/BiasAdd/ReadVariableOp^opt_8/Conv3D/ReadVariableOp^opt_9/BiasAdd/ReadVariableOp^opt_9/Conv3D/ReadVariableOp^pred/BiasAdd/ReadVariableOp^pred/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp28
fc_1/MatMul/ReadVariableOpfc_1/MatMul/ReadVariableOp2:
fc_3/BiasAdd/ReadVariableOpfc_3/BiasAdd/ReadVariableOp28
fc_3/MatMul/ReadVariableOpfc_3/MatMul/ReadVariableOp2>
gsc_11/BiasAdd/ReadVariableOpgsc_11/BiasAdd/ReadVariableOp2<
gsc_11/Conv3D/ReadVariableOpgsc_11/Conv3D/ReadVariableOp2>
gsc_12/BiasAdd/ReadVariableOpgsc_12/BiasAdd/ReadVariableOp2<
gsc_12/Conv3D/ReadVariableOpgsc_12/Conv3D/ReadVariableOp2<
gsc_2/BiasAdd/ReadVariableOpgsc_2/BiasAdd/ReadVariableOp2:
gsc_2/Conv3D/ReadVariableOpgsc_2/Conv3D/ReadVariableOp2<
gsc_3/BiasAdd/ReadVariableOpgsc_3/BiasAdd/ReadVariableOp2:
gsc_3/Conv3D/ReadVariableOpgsc_3/Conv3D/ReadVariableOp2<
gsc_5/BiasAdd/ReadVariableOpgsc_5/BiasAdd/ReadVariableOp2:
gsc_5/Conv3D/ReadVariableOpgsc_5/Conv3D/ReadVariableOp2<
gsc_6/BiasAdd/ReadVariableOpgsc_6/BiasAdd/ReadVariableOp2:
gsc_6/Conv3D/ReadVariableOpgsc_6/Conv3D/ReadVariableOp2<
gsc_8/BiasAdd/ReadVariableOpgsc_8/BiasAdd/ReadVariableOp2:
gsc_8/Conv3D/ReadVariableOpgsc_8/Conv3D/ReadVariableOp2<
gsc_9/BiasAdd/ReadVariableOpgsc_9/BiasAdd/ReadVariableOp2:
gsc_9/Conv3D/ReadVariableOpgsc_9/Conv3D/ReadVariableOp2@
merge_1/BiasAdd/ReadVariableOpmerge_1/BiasAdd/ReadVariableOp2>
merge_1/Conv3D/ReadVariableOpmerge_1/Conv3D/ReadVariableOp2@
merge_2/BiasAdd/ReadVariableOpmerge_2/BiasAdd/ReadVariableOp2>
merge_2/Conv3D/ReadVariableOpmerge_2/Conv3D/ReadVariableOp2@
merge_4/BiasAdd/ReadVariableOpmerge_4/BiasAdd/ReadVariableOp2>
merge_4/Conv3D/ReadVariableOpmerge_4/Conv3D/ReadVariableOp2@
merge_5/BiasAdd/ReadVariableOpmerge_5/BiasAdd/ReadVariableOp2>
merge_5/Conv3D/ReadVariableOpmerge_5/Conv3D/ReadVariableOp2@
merge_7/BiasAdd/ReadVariableOpmerge_7/BiasAdd/ReadVariableOp2>
merge_7/Conv3D/ReadVariableOpmerge_7/Conv3D/ReadVariableOp2@
merge_8/BiasAdd/ReadVariableOpmerge_8/BiasAdd/ReadVariableOp2>
merge_8/Conv3D/ReadVariableOpmerge_8/Conv3D/ReadVariableOp2>
opt_11/BiasAdd/ReadVariableOpopt_11/BiasAdd/ReadVariableOp2<
opt_11/Conv3D/ReadVariableOpopt_11/Conv3D/ReadVariableOp2>
opt_12/BiasAdd/ReadVariableOpopt_12/BiasAdd/ReadVariableOp2<
opt_12/Conv3D/ReadVariableOpopt_12/Conv3D/ReadVariableOp2<
opt_2/BiasAdd/ReadVariableOpopt_2/BiasAdd/ReadVariableOp2:
opt_2/Conv3D/ReadVariableOpopt_2/Conv3D/ReadVariableOp2<
opt_3/BiasAdd/ReadVariableOpopt_3/BiasAdd/ReadVariableOp2:
opt_3/Conv3D/ReadVariableOpopt_3/Conv3D/ReadVariableOp2<
opt_5/BiasAdd/ReadVariableOpopt_5/BiasAdd/ReadVariableOp2:
opt_5/Conv3D/ReadVariableOpopt_5/Conv3D/ReadVariableOp2<
opt_6/BiasAdd/ReadVariableOpopt_6/BiasAdd/ReadVariableOp2:
opt_6/Conv3D/ReadVariableOpopt_6/Conv3D/ReadVariableOp2<
opt_8/BiasAdd/ReadVariableOpopt_8/BiasAdd/ReadVariableOp2:
opt_8/Conv3D/ReadVariableOpopt_8/Conv3D/ReadVariableOp2<
opt_9/BiasAdd/ReadVariableOpopt_9/BiasAdd/ReadVariableOp2:
opt_9/Conv3D/ReadVariableOpopt_9/Conv3D/ReadVariableOp2:
pred/BiasAdd/ReadVariableOppred/BiasAdd/ReadVariableOp28
pred/MatMul/ReadVariableOppred/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
@__inference_opt_8_layer_call_and_return_conditional_losses_71561

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 b
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 f
IdentityIdentitySigmoid:y:0^NoOp*
T0*3
_output_shapes!
:?????????@88 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@88
 
_user_specified_nameinputs
?
?
@__inference_gsc_5_layer_call_and_return_conditional_losses_68990

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ppm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
?
A__inference_gsc_12_layer_call_and_return_conditional_losses_71681

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
?
^
B__inference_merge_3_layer_call_and_return_conditional_losses_68838

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
??
?$
C__inference_cnn_base_layer_call_and_return_conditional_losses_70952

inputsB
$opt_2_conv3d_readvariableop_resource:3
%opt_2_biasadd_readvariableop_resource:B
$gsc_2_conv3d_readvariableop_resource:3
%gsc_2_biasadd_readvariableop_resource:B
$opt_3_conv3d_readvariableop_resource:3
%opt_3_biasadd_readvariableop_resource:B
$gsc_3_conv3d_readvariableop_resource:3
%gsc_3_biasadd_readvariableop_resource:B
$opt_5_conv3d_readvariableop_resource:3
%opt_5_biasadd_readvariableop_resource:B
$gsc_5_conv3d_readvariableop_resource:3
%gsc_5_biasadd_readvariableop_resource:B
$opt_6_conv3d_readvariableop_resource:3
%opt_6_biasadd_readvariableop_resource:B
$gsc_6_conv3d_readvariableop_resource:3
%gsc_6_biasadd_readvariableop_resource:B
$opt_8_conv3d_readvariableop_resource: 3
%opt_8_biasadd_readvariableop_resource: B
$gsc_8_conv3d_readvariableop_resource: 3
%gsc_8_biasadd_readvariableop_resource: B
$opt_9_conv3d_readvariableop_resource:  3
%opt_9_biasadd_readvariableop_resource: B
$gsc_9_conv3d_readvariableop_resource:  3
%gsc_9_biasadd_readvariableop_resource: C
%opt_11_conv3d_readvariableop_resource:  4
&opt_11_biasadd_readvariableop_resource: C
%gsc_11_conv3d_readvariableop_resource:  4
&gsc_11_biasadd_readvariableop_resource: C
%opt_12_conv3d_readvariableop_resource:  4
&opt_12_biasadd_readvariableop_resource: C
%gsc_12_conv3d_readvariableop_resource:  4
&gsc_12_biasadd_readvariableop_resource: D
&merge_1_conv3d_readvariableop_resource: @5
'merge_1_biasadd_readvariableop_resource:@D
&merge_2_conv3d_readvariableop_resource:@@5
'merge_2_biasadd_readvariableop_resource:@D
&merge_4_conv3d_readvariableop_resource:@@5
'merge_4_biasadd_readvariableop_resource:@D
&merge_5_conv3d_readvariableop_resource:@@5
'merge_5_biasadd_readvariableop_resource:@E
&merge_7_conv3d_readvariableop_resource:@?6
'merge_7_biasadd_readvariableop_resource:	?F
&merge_8_conv3d_readvariableop_resource:??6
'merge_8_biasadd_readvariableop_resource:	?7
#fc_1_matmul_readvariableop_resource:
??3
$fc_1_biasadd_readvariableop_resource:	?6
#fc_3_matmul_readvariableop_resource:	? 2
$fc_3_biasadd_readvariableop_resource: 5
#pred_matmul_readvariableop_resource: 2
$pred_biasadd_readvariableop_resource:
identity??fc_1/BiasAdd/ReadVariableOp?fc_1/MatMul/ReadVariableOp?fc_3/BiasAdd/ReadVariableOp?fc_3/MatMul/ReadVariableOp?gsc_11/BiasAdd/ReadVariableOp?gsc_11/Conv3D/ReadVariableOp?gsc_12/BiasAdd/ReadVariableOp?gsc_12/Conv3D/ReadVariableOp?gsc_2/BiasAdd/ReadVariableOp?gsc_2/Conv3D/ReadVariableOp?gsc_3/BiasAdd/ReadVariableOp?gsc_3/Conv3D/ReadVariableOp?gsc_5/BiasAdd/ReadVariableOp?gsc_5/Conv3D/ReadVariableOp?gsc_6/BiasAdd/ReadVariableOp?gsc_6/Conv3D/ReadVariableOp?gsc_8/BiasAdd/ReadVariableOp?gsc_8/Conv3D/ReadVariableOp?gsc_9/BiasAdd/ReadVariableOp?gsc_9/Conv3D/ReadVariableOp?merge_1/BiasAdd/ReadVariableOp?merge_1/Conv3D/ReadVariableOp?merge_2/BiasAdd/ReadVariableOp?merge_2/Conv3D/ReadVariableOp?merge_4/BiasAdd/ReadVariableOp?merge_4/Conv3D/ReadVariableOp?merge_5/BiasAdd/ReadVariableOp?merge_5/Conv3D/ReadVariableOp?merge_7/BiasAdd/ReadVariableOp?merge_7/Conv3D/ReadVariableOp?merge_8/BiasAdd/ReadVariableOp?merge_8/Conv3D/ReadVariableOp?opt_11/BiasAdd/ReadVariableOp?opt_11/Conv3D/ReadVariableOp?opt_12/BiasAdd/ReadVariableOp?opt_12/Conv3D/ReadVariableOp?opt_2/BiasAdd/ReadVariableOp?opt_2/Conv3D/ReadVariableOp?opt_3/BiasAdd/ReadVariableOp?opt_3/Conv3D/ReadVariableOp?opt_5/BiasAdd/ReadVariableOp?opt_5/Conv3D/ReadVariableOp?opt_6/BiasAdd/ReadVariableOp?opt_6/Conv3D/ReadVariableOp?opt_8/BiasAdd/ReadVariableOp?opt_8/Conv3D/ReadVariableOp?opt_9/BiasAdd/ReadVariableOp?opt_9/Conv3D/ReadVariableOp?pred/BiasAdd/ReadVariableOp?pred/MatMul/ReadVariableOpj
opt_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       l
opt_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        l
opt_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
opt_1/strided_sliceStridedSliceinputs"opt_1/strided_slice/stack:output:0$opt_1/strided_slice/stack_1:output:0$opt_1/strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*
ellipsis_mask*
end_maskj
gsc_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        l
gsc_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       l
gsc_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
gsc_1/strided_sliceStridedSliceinputs"gsc_1/strided_slice/stack:output:0$gsc_1/strided_slice/stack_1:output:0$gsc_1/strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*

begin_mask*
ellipsis_mask?
opt_2/Conv3D/ReadVariableOpReadVariableOp$opt_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
opt_2/Conv3DConv3Dopt_1/strided_slice:output:0#opt_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
~
opt_2/BiasAdd/ReadVariableOpReadVariableOp%opt_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
opt_2/BiasAddBiasAddopt_2/Conv3D:output:0$opt_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??j

opt_2/ReluReluopt_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
gsc_2/Conv3D/ReadVariableOpReadVariableOp$gsc_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
gsc_2/Conv3DConv3Dgsc_1/strided_slice:output:0#gsc_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
~
gsc_2/BiasAdd/ReadVariableOpReadVariableOp%gsc_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
gsc_2/BiasAddBiasAddgsc_2/Conv3D:output:0$gsc_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??j

gsc_2/ReluRelugsc_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
opt_3/Conv3D/ReadVariableOpReadVariableOp$opt_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
opt_3/Conv3DConv3Dopt_2/Relu:activations:0#opt_3/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
~
opt_3/BiasAdd/ReadVariableOpReadVariableOp%opt_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
opt_3/BiasAddBiasAddopt_3/Conv3D:output:0$opt_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??j

opt_3/ReluReluopt_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
gsc_3/Conv3D/ReadVariableOpReadVariableOp$gsc_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
gsc_3/Conv3DConv3Dgsc_2/Relu:activations:0#gsc_3/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
~
gsc_3/BiasAdd/ReadVariableOpReadVariableOp%gsc_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
gsc_3/BiasAddBiasAddgsc_3/Conv3D:output:0$gsc_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??j

gsc_3/ReluRelugsc_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
opt_4/MaxPool3D	MaxPool3Dopt_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????@pp*
ksize	
*
paddingVALID*
strides	
?
gsc_4/MaxPool3D	MaxPool3Dgsc_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????@pp*
ksize	
*
paddingVALID*
strides	
?
opt_5/Conv3D/ReadVariableOpReadVariableOp$opt_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
opt_5/Conv3DConv3Dopt_4/MaxPool3D:output:0#opt_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
~
opt_5/BiasAdd/ReadVariableOpReadVariableOp%opt_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
opt_5/BiasAddBiasAddopt_5/Conv3D:output:0$opt_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pph

opt_5/ReluReluopt_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
gsc_5/Conv3D/ReadVariableOpReadVariableOp$gsc_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
gsc_5/Conv3DConv3Dgsc_4/MaxPool3D:output:0#gsc_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
~
gsc_5/BiasAdd/ReadVariableOpReadVariableOp%gsc_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
gsc_5/BiasAddBiasAddgsc_5/Conv3D:output:0$gsc_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pph

gsc_5/ReluRelugsc_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
opt_6/Conv3D/ReadVariableOpReadVariableOp$opt_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
opt_6/Conv3DConv3Dopt_5/Relu:activations:0#opt_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
~
opt_6/BiasAdd/ReadVariableOpReadVariableOp%opt_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
opt_6/BiasAddBiasAddopt_6/Conv3D:output:0$opt_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pph

opt_6/ReluReluopt_6/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
gsc_6/Conv3D/ReadVariableOpReadVariableOp$gsc_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
gsc_6/Conv3DConv3Dgsc_5/Relu:activations:0#gsc_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
~
gsc_6/BiasAdd/ReadVariableOpReadVariableOp%gsc_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
gsc_6/BiasAddBiasAddgsc_6/Conv3D:output:0$gsc_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pph

gsc_6/ReluRelugsc_6/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
opt_7/MaxPool3D	MaxPool3Dopt_6/Relu:activations:0*
T0*3
_output_shapes!
:?????????@88*
ksize	
*
paddingVALID*
strides	
?
gsc_7/MaxPool3D	MaxPool3Dgsc_6/Relu:activations:0*
T0*3
_output_shapes!
:?????????@88*
ksize	
*
paddingVALID*
strides	
?
opt_8/Conv3D/ReadVariableOpReadVariableOp$opt_8_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0?
opt_8/Conv3DConv3Dopt_7/MaxPool3D:output:0#opt_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
~
opt_8/BiasAdd/ReadVariableOpReadVariableOp%opt_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
opt_8/BiasAddBiasAddopt_8/Conv3D:output:0$opt_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 n
opt_8/SigmoidSigmoidopt_8/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
gsc_8/Conv3D/ReadVariableOpReadVariableOp$gsc_8_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0?
gsc_8/Conv3DConv3Dgsc_7/MaxPool3D:output:0#gsc_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
~
gsc_8/BiasAdd/ReadVariableOpReadVariableOp%gsc_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
gsc_8/BiasAddBiasAddgsc_8/Conv3D:output:0$gsc_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 h

gsc_8/ReluRelugsc_8/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
opt_9/Conv3D/ReadVariableOpReadVariableOp$opt_9_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
opt_9/Conv3DConv3Dopt_8/Sigmoid:y:0#opt_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
~
opt_9/BiasAdd/ReadVariableOpReadVariableOp%opt_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
opt_9/BiasAddBiasAddopt_9/Conv3D:output:0$opt_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 n
opt_9/SigmoidSigmoidopt_9/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
gsc_9/Conv3D/ReadVariableOpReadVariableOp$gsc_9_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
gsc_9/Conv3DConv3Dgsc_8/Relu:activations:0#gsc_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
~
gsc_9/BiasAdd/ReadVariableOpReadVariableOp%gsc_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
gsc_9/BiasAddBiasAddgsc_9/Conv3D:output:0$gsc_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 h

gsc_9/ReluRelugsc_9/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
opt_10/MaxPool3D	MaxPool3Dopt_9/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?
gsc_10/MaxPool3D	MaxPool3Dgsc_9/Relu:activations:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?
opt_11/Conv3D/ReadVariableOpReadVariableOp%opt_11_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
opt_11/Conv3DConv3Dopt_10/MaxPool3D:output:0$opt_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
opt_11/BiasAdd/ReadVariableOpReadVariableOp&opt_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
opt_11/BiasAddBiasAddopt_11/Conv3D:output:0%opt_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ p
opt_11/SigmoidSigmoidopt_11/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
gsc_11/Conv3D/ReadVariableOpReadVariableOp%gsc_11_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
gsc_11/Conv3DConv3Dgsc_10/MaxPool3D:output:0$gsc_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
gsc_11/BiasAdd/ReadVariableOpReadVariableOp&gsc_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
gsc_11/BiasAddBiasAddgsc_11/Conv3D:output:0%gsc_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ j
gsc_11/ReluRelugsc_11/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
opt_12/Conv3D/ReadVariableOpReadVariableOp%opt_12_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
opt_12/Conv3DConv3Dopt_11/Sigmoid:y:0$opt_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
opt_12/BiasAdd/ReadVariableOpReadVariableOp&opt_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
opt_12/BiasAddBiasAddopt_12/Conv3D:output:0%opt_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ p
opt_12/SigmoidSigmoidopt_12/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
gsc_12/Conv3D/ReadVariableOpReadVariableOp%gsc_12_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
gsc_12/Conv3DConv3Dgsc_11/Relu:activations:0$gsc_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
gsc_12/BiasAdd/ReadVariableOpReadVariableOp&gsc_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
gsc_12/BiasAddBiasAddgsc_12/Conv3D:output:0%gsc_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ j
gsc_12/ReluRelugsc_12/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
gsc_13/MaxPool3D	MaxPool3Dgsc_12/Relu:activations:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?
opt_13/MaxPool3D	MaxPool3Dopt_12/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?

fuse_1/mulMulgsc_13/MaxPool3D:output:0opt_13/MaxPool3D:output:0*
T0*3
_output_shapes!
:?????????@ ?
fuse_2/MaxPool3D	MaxPool3Dfuse_1/mul:z:0*
T0*3
_output_shapes!
:????????? *
ksize	
*
paddingVALID*
strides	
?
merge_1/Conv3D/ReadVariableOpReadVariableOp&merge_1_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0?
merge_1/Conv3DConv3Dfuse_2/MaxPool3D:output:0%merge_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
merge_1/BiasAdd/ReadVariableOpReadVariableOp'merge_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
merge_1/BiasAddBiasAddmerge_1/Conv3D:output:0&merge_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@l
merge_1/ReluRelumerge_1/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
merge_2/Conv3D/ReadVariableOpReadVariableOp&merge_2_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
merge_2/Conv3DConv3Dmerge_1/Relu:activations:0%merge_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
merge_2/BiasAdd/ReadVariableOpReadVariableOp'merge_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
merge_2/BiasAddBiasAddmerge_2/Conv3D:output:0&merge_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@l
merge_2/ReluRelumerge_2/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
merge_3/MaxPool3D	MaxPool3Dmerge_2/Relu:activations:0*
T0*3
_output_shapes!
:?????????@*
ksize	
*
paddingVALID*
strides	
?
merge_4/Conv3D/ReadVariableOpReadVariableOp&merge_4_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
merge_4/Conv3DConv3Dmerge_3/MaxPool3D:output:0%merge_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
merge_4/BiasAdd/ReadVariableOpReadVariableOp'merge_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
merge_4/BiasAddBiasAddmerge_4/Conv3D:output:0&merge_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@l
merge_4/ReluRelumerge_4/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
merge_5/Conv3D/ReadVariableOpReadVariableOp&merge_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
merge_5/Conv3DConv3Dmerge_4/Relu:activations:0%merge_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
merge_5/BiasAdd/ReadVariableOpReadVariableOp'merge_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
merge_5/BiasAddBiasAddmerge_5/Conv3D:output:0&merge_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@l
merge_5/ReluRelumerge_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
merge_6/MaxPool3D	MaxPool3Dmerge_5/Relu:activations:0*
T0*3
_output_shapes!
:?????????@*
ksize	
*
paddingVALID*
strides	
?
merge_7/Conv3D/ReadVariableOpReadVariableOp&merge_7_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype0?
merge_7/Conv3DConv3Dmerge_6/MaxPool3D:output:0%merge_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
?
merge_7/BiasAdd/ReadVariableOpReadVariableOp'merge_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
merge_7/BiasAddBiasAddmerge_7/Conv3D:output:0&merge_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????m
merge_7/ReluRelumerge_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :???????????
merge_8/Conv3D/ReadVariableOpReadVariableOp&merge_8_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype0?
merge_8/Conv3DConv3Dmerge_7/Relu:activations:0%merge_8/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
?
merge_8/BiasAdd/ReadVariableOpReadVariableOp'merge_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
merge_8/BiasAddBiasAddmerge_8/Conv3D:output:0&merge_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????m
merge_8/ReluRelumerge_8/BiasAdd:output:0*
T0*4
_output_shapes"
 :???????????
merge_9/MaxPool3D	MaxPool3Dmerge_8/Relu:activations:0*
T0*4
_output_shapes"
 :??????????*
ksize	
*
paddingVALID*
strides	
[

flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   {
flat/ReshapeReshapemerge_9/MaxPool3D:output:0flat/Const:output:0*
T0*(
_output_shapes
:???????????
fc_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
fc_1/MatMulMatMulflat/Reshape:output:0"fc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????}
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????[
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????e
fc_2/IdentityIdentityfc_1/Relu:activations:0*
T0*(
_output_shapes
:??????????
fc_3/MatMul/ReadVariableOpReadVariableOp#fc_3_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
fc_3/MatMulMatMulfc_2/Identity:output:0"fc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? |
fc_3/BiasAdd/ReadVariableOpReadVariableOp$fc_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
fc_3/BiasAddBiasAddfc_3/MatMul:product:0#fc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? Z
	fc_3/ReluRelufc_3/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ~
pred/MatMul/ReadVariableOpReadVariableOp#pred_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
pred/MatMulMatMulfc_3/Relu:activations:0"pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
pred/BiasAdd/ReadVariableOpReadVariableOp$pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
pred/BiasAddBiasAddpred/MatMul:product:0#pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
pred/SigmoidSigmoidpred/BiasAdd:output:0*
T0*'
_output_shapes
:?????????_
IdentityIdentitypred/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^fc_1/BiasAdd/ReadVariableOp^fc_1/MatMul/ReadVariableOp^fc_3/BiasAdd/ReadVariableOp^fc_3/MatMul/ReadVariableOp^gsc_11/BiasAdd/ReadVariableOp^gsc_11/Conv3D/ReadVariableOp^gsc_12/BiasAdd/ReadVariableOp^gsc_12/Conv3D/ReadVariableOp^gsc_2/BiasAdd/ReadVariableOp^gsc_2/Conv3D/ReadVariableOp^gsc_3/BiasAdd/ReadVariableOp^gsc_3/Conv3D/ReadVariableOp^gsc_5/BiasAdd/ReadVariableOp^gsc_5/Conv3D/ReadVariableOp^gsc_6/BiasAdd/ReadVariableOp^gsc_6/Conv3D/ReadVariableOp^gsc_8/BiasAdd/ReadVariableOp^gsc_8/Conv3D/ReadVariableOp^gsc_9/BiasAdd/ReadVariableOp^gsc_9/Conv3D/ReadVariableOp^merge_1/BiasAdd/ReadVariableOp^merge_1/Conv3D/ReadVariableOp^merge_2/BiasAdd/ReadVariableOp^merge_2/Conv3D/ReadVariableOp^merge_4/BiasAdd/ReadVariableOp^merge_4/Conv3D/ReadVariableOp^merge_5/BiasAdd/ReadVariableOp^merge_5/Conv3D/ReadVariableOp^merge_7/BiasAdd/ReadVariableOp^merge_7/Conv3D/ReadVariableOp^merge_8/BiasAdd/ReadVariableOp^merge_8/Conv3D/ReadVariableOp^opt_11/BiasAdd/ReadVariableOp^opt_11/Conv3D/ReadVariableOp^opt_12/BiasAdd/ReadVariableOp^opt_12/Conv3D/ReadVariableOp^opt_2/BiasAdd/ReadVariableOp^opt_2/Conv3D/ReadVariableOp^opt_3/BiasAdd/ReadVariableOp^opt_3/Conv3D/ReadVariableOp^opt_5/BiasAdd/ReadVariableOp^opt_5/Conv3D/ReadVariableOp^opt_6/BiasAdd/ReadVariableOp^opt_6/Conv3D/ReadVariableOp^opt_8/BiasAdd/ReadVariableOp^opt_8/Conv3D/ReadVariableOp^opt_9/BiasAdd/ReadVariableOp^opt_9/Conv3D/ReadVariableOp^pred/BiasAdd/ReadVariableOp^pred/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp28
fc_1/MatMul/ReadVariableOpfc_1/MatMul/ReadVariableOp2:
fc_3/BiasAdd/ReadVariableOpfc_3/BiasAdd/ReadVariableOp28
fc_3/MatMul/ReadVariableOpfc_3/MatMul/ReadVariableOp2>
gsc_11/BiasAdd/ReadVariableOpgsc_11/BiasAdd/ReadVariableOp2<
gsc_11/Conv3D/ReadVariableOpgsc_11/Conv3D/ReadVariableOp2>
gsc_12/BiasAdd/ReadVariableOpgsc_12/BiasAdd/ReadVariableOp2<
gsc_12/Conv3D/ReadVariableOpgsc_12/Conv3D/ReadVariableOp2<
gsc_2/BiasAdd/ReadVariableOpgsc_2/BiasAdd/ReadVariableOp2:
gsc_2/Conv3D/ReadVariableOpgsc_2/Conv3D/ReadVariableOp2<
gsc_3/BiasAdd/ReadVariableOpgsc_3/BiasAdd/ReadVariableOp2:
gsc_3/Conv3D/ReadVariableOpgsc_3/Conv3D/ReadVariableOp2<
gsc_5/BiasAdd/ReadVariableOpgsc_5/BiasAdd/ReadVariableOp2:
gsc_5/Conv3D/ReadVariableOpgsc_5/Conv3D/ReadVariableOp2<
gsc_6/BiasAdd/ReadVariableOpgsc_6/BiasAdd/ReadVariableOp2:
gsc_6/Conv3D/ReadVariableOpgsc_6/Conv3D/ReadVariableOp2<
gsc_8/BiasAdd/ReadVariableOpgsc_8/BiasAdd/ReadVariableOp2:
gsc_8/Conv3D/ReadVariableOpgsc_8/Conv3D/ReadVariableOp2<
gsc_9/BiasAdd/ReadVariableOpgsc_9/BiasAdd/ReadVariableOp2:
gsc_9/Conv3D/ReadVariableOpgsc_9/Conv3D/ReadVariableOp2@
merge_1/BiasAdd/ReadVariableOpmerge_1/BiasAdd/ReadVariableOp2>
merge_1/Conv3D/ReadVariableOpmerge_1/Conv3D/ReadVariableOp2@
merge_2/BiasAdd/ReadVariableOpmerge_2/BiasAdd/ReadVariableOp2>
merge_2/Conv3D/ReadVariableOpmerge_2/Conv3D/ReadVariableOp2@
merge_4/BiasAdd/ReadVariableOpmerge_4/BiasAdd/ReadVariableOp2>
merge_4/Conv3D/ReadVariableOpmerge_4/Conv3D/ReadVariableOp2@
merge_5/BiasAdd/ReadVariableOpmerge_5/BiasAdd/ReadVariableOp2>
merge_5/Conv3D/ReadVariableOpmerge_5/Conv3D/ReadVariableOp2@
merge_7/BiasAdd/ReadVariableOpmerge_7/BiasAdd/ReadVariableOp2>
merge_7/Conv3D/ReadVariableOpmerge_7/Conv3D/ReadVariableOp2@
merge_8/BiasAdd/ReadVariableOpmerge_8/BiasAdd/ReadVariableOp2>
merge_8/Conv3D/ReadVariableOpmerge_8/Conv3D/ReadVariableOp2>
opt_11/BiasAdd/ReadVariableOpopt_11/BiasAdd/ReadVariableOp2<
opt_11/Conv3D/ReadVariableOpopt_11/Conv3D/ReadVariableOp2>
opt_12/BiasAdd/ReadVariableOpopt_12/BiasAdd/ReadVariableOp2<
opt_12/Conv3D/ReadVariableOpopt_12/Conv3D/ReadVariableOp2<
opt_2/BiasAdd/ReadVariableOpopt_2/BiasAdd/ReadVariableOp2:
opt_2/Conv3D/ReadVariableOpopt_2/Conv3D/ReadVariableOp2<
opt_3/BiasAdd/ReadVariableOpopt_3/BiasAdd/ReadVariableOp2:
opt_3/Conv3D/ReadVariableOpopt_3/Conv3D/ReadVariableOp2<
opt_5/BiasAdd/ReadVariableOpopt_5/BiasAdd/ReadVariableOp2:
opt_5/Conv3D/ReadVariableOpopt_5/Conv3D/ReadVariableOp2<
opt_6/BiasAdd/ReadVariableOpopt_6/BiasAdd/ReadVariableOp2:
opt_6/Conv3D/ReadVariableOpopt_6/Conv3D/ReadVariableOp2<
opt_8/BiasAdd/ReadVariableOpopt_8/BiasAdd/ReadVariableOp2:
opt_8/Conv3D/ReadVariableOpopt_8/Conv3D/ReadVariableOp2<
opt_9/BiasAdd/ReadVariableOpopt_9/BiasAdd/ReadVariableOp2:
opt_9/Conv3D/ReadVariableOpopt_9/Conv3D/ReadVariableOp2:
pred/BiasAdd/ReadVariableOppred/BiasAdd/ReadVariableOp28
pred/MatMul/ReadVariableOppred/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
'__inference_merge_2_layer_call_fn_71772

inputs%
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_2_layer_call_and_return_conditional_losses_69209{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
]
?__inference_fc_2_layer_call_and_return_conditional_losses_69316

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
C__inference_cnn_base_layer_call_and_return_conditional_losses_69353

inputs)
opt_2_68904:
opt_2_68906:)
gsc_2_68921:
gsc_2_68923:)
opt_3_68938:
opt_3_68940:)
gsc_3_68955:
gsc_3_68957:)
opt_5_68974:
opt_5_68976:)
gsc_5_68991:
gsc_5_68993:)
opt_6_69008:
opt_6_69010:)
gsc_6_69025:
gsc_6_69027:)
opt_8_69044: 
opt_8_69046: )
gsc_8_69061: 
gsc_8_69063: )
opt_9_69078:  
opt_9_69080: )
gsc_9_69095:  
gsc_9_69097: *
opt_11_69114:  
opt_11_69116: *
gsc_11_69131:  
gsc_11_69133: *
opt_12_69148:  
opt_12_69150: *
gsc_12_69165:  
gsc_12_69167: +
merge_1_69193: @
merge_1_69195:@+
merge_2_69210:@@
merge_2_69212:@+
merge_4_69228:@@
merge_4_69230:@+
merge_5_69245:@@
merge_5_69247:@,
merge_7_69263:@?
merge_7_69265:	?-
merge_8_69280:??
merge_8_69282:	?

fc_1_69306:
??

fc_1_69308:	?

fc_3_69330:	? 

fc_3_69332: 

pred_69347: 

pred_69349:
identity??fc_1/StatefulPartitionedCall?fc_3/StatefulPartitionedCall?gsc_11/StatefulPartitionedCall?gsc_12/StatefulPartitionedCall?gsc_2/StatefulPartitionedCall?gsc_3/StatefulPartitionedCall?gsc_5/StatefulPartitionedCall?gsc_6/StatefulPartitionedCall?gsc_8/StatefulPartitionedCall?gsc_9/StatefulPartitionedCall?merge_1/StatefulPartitionedCall?merge_2/StatefulPartitionedCall?merge_4/StatefulPartitionedCall?merge_5/StatefulPartitionedCall?merge_7/StatefulPartitionedCall?merge_8/StatefulPartitionedCall?opt_11/StatefulPartitionedCall?opt_12/StatefulPartitionedCall?opt_2/StatefulPartitionedCall?opt_3/StatefulPartitionedCall?opt_5/StatefulPartitionedCall?opt_6/StatefulPartitionedCall?opt_8/StatefulPartitionedCall?opt_9/StatefulPartitionedCall?pred/StatefulPartitionedCall?
opt_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_1_layer_call_and_return_conditional_losses_68880?
gsc_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_1_layer_call_and_return_conditional_losses_68890?
opt_2/StatefulPartitionedCallStatefulPartitionedCallopt_1/PartitionedCall:output:0opt_2_68904opt_2_68906*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_2_layer_call_and_return_conditional_losses_68903?
gsc_2/StatefulPartitionedCallStatefulPartitionedCallgsc_1/PartitionedCall:output:0gsc_2_68921gsc_2_68923*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_2_layer_call_and_return_conditional_losses_68920?
opt_3/StatefulPartitionedCallStatefulPartitionedCall&opt_2/StatefulPartitionedCall:output:0opt_3_68938opt_3_68940*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_3_layer_call_and_return_conditional_losses_68937?
gsc_3/StatefulPartitionedCallStatefulPartitionedCall&gsc_2/StatefulPartitionedCall:output:0gsc_3_68955gsc_3_68957*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_3_layer_call_and_return_conditional_losses_68954?
opt_4/PartitionedCallPartitionedCall&opt_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_4_layer_call_and_return_conditional_losses_68742?
gsc_4/PartitionedCallPartitionedCall&gsc_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_4_layer_call_and_return_conditional_losses_68730?
opt_5/StatefulPartitionedCallStatefulPartitionedCallopt_4/PartitionedCall:output:0opt_5_68974opt_5_68976*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_5_layer_call_and_return_conditional_losses_68973?
gsc_5/StatefulPartitionedCallStatefulPartitionedCallgsc_4/PartitionedCall:output:0gsc_5_68991gsc_5_68993*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_5_layer_call_and_return_conditional_losses_68990?
opt_6/StatefulPartitionedCallStatefulPartitionedCall&opt_5/StatefulPartitionedCall:output:0opt_6_69008opt_6_69010*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_6_layer_call_and_return_conditional_losses_69007?
gsc_6/StatefulPartitionedCallStatefulPartitionedCall&gsc_5/StatefulPartitionedCall:output:0gsc_6_69025gsc_6_69027*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_6_layer_call_and_return_conditional_losses_69024?
opt_7/PartitionedCallPartitionedCall&opt_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_7_layer_call_and_return_conditional_losses_68766?
gsc_7/PartitionedCallPartitionedCall&gsc_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_7_layer_call_and_return_conditional_losses_68754?
opt_8/StatefulPartitionedCallStatefulPartitionedCallopt_7/PartitionedCall:output:0opt_8_69044opt_8_69046*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_8_layer_call_and_return_conditional_losses_69043?
gsc_8/StatefulPartitionedCallStatefulPartitionedCallgsc_7/PartitionedCall:output:0gsc_8_69061gsc_8_69063*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_8_layer_call_and_return_conditional_losses_69060?
opt_9/StatefulPartitionedCallStatefulPartitionedCall&opt_8/StatefulPartitionedCall:output:0opt_9_69078opt_9_69080*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_9_layer_call_and_return_conditional_losses_69077?
gsc_9/StatefulPartitionedCallStatefulPartitionedCall&gsc_8/StatefulPartitionedCall:output:0gsc_9_69095gsc_9_69097*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_9_layer_call_and_return_conditional_losses_69094?
opt_10/PartitionedCallPartitionedCall&opt_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_10_layer_call_and_return_conditional_losses_68790?
gsc_10/PartitionedCallPartitionedCall&gsc_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_10_layer_call_and_return_conditional_losses_68778?
opt_11/StatefulPartitionedCallStatefulPartitionedCallopt_10/PartitionedCall:output:0opt_11_69114opt_11_69116*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_11_layer_call_and_return_conditional_losses_69113?
gsc_11/StatefulPartitionedCallStatefulPartitionedCallgsc_10/PartitionedCall:output:0gsc_11_69131gsc_11_69133*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_11_layer_call_and_return_conditional_losses_69130?
opt_12/StatefulPartitionedCallStatefulPartitionedCall'opt_11/StatefulPartitionedCall:output:0opt_12_69148opt_12_69150*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_12_layer_call_and_return_conditional_losses_69147?
gsc_12/StatefulPartitionedCallStatefulPartitionedCall'gsc_11/StatefulPartitionedCall:output:0gsc_12_69165gsc_12_69167*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_12_layer_call_and_return_conditional_losses_69164?
gsc_13/PartitionedCallPartitionedCall'gsc_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_13_layer_call_and_return_conditional_losses_68802?
opt_13/PartitionedCallPartitionedCall'opt_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_13_layer_call_and_return_conditional_losses_68814?
fuse_1/PartitionedCallPartitionedCallgsc_13/PartitionedCall:output:0opt_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_fuse_1_layer_call_and_return_conditional_losses_69178?
fuse_2/PartitionedCallPartitionedCallfuse_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_fuse_2_layer_call_and_return_conditional_losses_68826?
merge_1/StatefulPartitionedCallStatefulPartitionedCallfuse_2/PartitionedCall:output:0merge_1_69193merge_1_69195*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_1_layer_call_and_return_conditional_losses_69192?
merge_2/StatefulPartitionedCallStatefulPartitionedCall(merge_1/StatefulPartitionedCall:output:0merge_2_69210merge_2_69212*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_2_layer_call_and_return_conditional_losses_69209?
merge_3/PartitionedCallPartitionedCall(merge_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_3_layer_call_and_return_conditional_losses_68838?
merge_4/StatefulPartitionedCallStatefulPartitionedCall merge_3/PartitionedCall:output:0merge_4_69228merge_4_69230*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_4_layer_call_and_return_conditional_losses_69227?
merge_5/StatefulPartitionedCallStatefulPartitionedCall(merge_4/StatefulPartitionedCall:output:0merge_5_69245merge_5_69247*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_5_layer_call_and_return_conditional_losses_69244?
merge_6/PartitionedCallPartitionedCall(merge_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_6_layer_call_and_return_conditional_losses_68850?
merge_7/StatefulPartitionedCallStatefulPartitionedCall merge_6/PartitionedCall:output:0merge_7_69263merge_7_69265*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_7_layer_call_and_return_conditional_losses_69262?
merge_8/StatefulPartitionedCallStatefulPartitionedCall(merge_7/StatefulPartitionedCall:output:0merge_8_69280merge_8_69282*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_8_layer_call_and_return_conditional_losses_69279?
merge_9/PartitionedCallPartitionedCall(merge_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_9_layer_call_and_return_conditional_losses_68862?
flat/PartitionedCallPartitionedCall merge_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_flat_layer_call_and_return_conditional_losses_69292?
fc_1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0
fc_1_69306
fc_1_69308*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_1_layer_call_and_return_conditional_losses_69305?
fc_2/PartitionedCallPartitionedCall%fc_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_2_layer_call_and_return_conditional_losses_69316?
fc_3/StatefulPartitionedCallStatefulPartitionedCallfc_2/PartitionedCall:output:0
fc_3_69330
fc_3_69332*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_3_layer_call_and_return_conditional_losses_69329?
pred/StatefulPartitionedCallStatefulPartitionedCall%fc_3/StatefulPartitionedCall:output:0
pred_69347
pred_69349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_pred_layer_call_and_return_conditional_losses_69346t
IdentityIdentity%pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^fc_1/StatefulPartitionedCall^fc_3/StatefulPartitionedCall^gsc_11/StatefulPartitionedCall^gsc_12/StatefulPartitionedCall^gsc_2/StatefulPartitionedCall^gsc_3/StatefulPartitionedCall^gsc_5/StatefulPartitionedCall^gsc_6/StatefulPartitionedCall^gsc_8/StatefulPartitionedCall^gsc_9/StatefulPartitionedCall ^merge_1/StatefulPartitionedCall ^merge_2/StatefulPartitionedCall ^merge_4/StatefulPartitionedCall ^merge_5/StatefulPartitionedCall ^merge_7/StatefulPartitionedCall ^merge_8/StatefulPartitionedCall^opt_11/StatefulPartitionedCall^opt_12/StatefulPartitionedCall^opt_2/StatefulPartitionedCall^opt_3/StatefulPartitionedCall^opt_5/StatefulPartitionedCall^opt_6/StatefulPartitionedCall^opt_8/StatefulPartitionedCall^opt_9/StatefulPartitionedCall^pred/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2@
gsc_11/StatefulPartitionedCallgsc_11/StatefulPartitionedCall2@
gsc_12/StatefulPartitionedCallgsc_12/StatefulPartitionedCall2>
gsc_2/StatefulPartitionedCallgsc_2/StatefulPartitionedCall2>
gsc_3/StatefulPartitionedCallgsc_3/StatefulPartitionedCall2>
gsc_5/StatefulPartitionedCallgsc_5/StatefulPartitionedCall2>
gsc_6/StatefulPartitionedCallgsc_6/StatefulPartitionedCall2>
gsc_8/StatefulPartitionedCallgsc_8/StatefulPartitionedCall2>
gsc_9/StatefulPartitionedCallgsc_9/StatefulPartitionedCall2B
merge_1/StatefulPartitionedCallmerge_1/StatefulPartitionedCall2B
merge_2/StatefulPartitionedCallmerge_2/StatefulPartitionedCall2B
merge_4/StatefulPartitionedCallmerge_4/StatefulPartitionedCall2B
merge_5/StatefulPartitionedCallmerge_5/StatefulPartitionedCall2B
merge_7/StatefulPartitionedCallmerge_7/StatefulPartitionedCall2B
merge_8/StatefulPartitionedCallmerge_8/StatefulPartitionedCall2@
opt_11/StatefulPartitionedCallopt_11/StatefulPartitionedCall2@
opt_12/StatefulPartitionedCallopt_12/StatefulPartitionedCall2>
opt_2/StatefulPartitionedCallopt_2/StatefulPartitionedCall2>
opt_3/StatefulPartitionedCallopt_3/StatefulPartitionedCall2>
opt_5/StatefulPartitionedCallopt_5/StatefulPartitionedCall2>
opt_6/StatefulPartitionedCallopt_6/StatefulPartitionedCall2>
opt_8/StatefulPartitionedCallopt_8/StatefulPartitionedCall2>
opt_9/StatefulPartitionedCallopt_9/StatefulPartitionedCall2<
pred/StatefulPartitionedCallpred/StatefulPartitionedCall:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
\
@__inference_gsc_1_layer_call_and_return_conditional_losses_69758

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*

begin_mask*
ellipsis_maskl
IdentityIdentitystrided_slice:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?

?
?__inference_fc_3_layer_call_and_return_conditional_losses_69329

inputs1
matmul_readvariableop_resource:	? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_gsc_2_layer_call_and_return_conditional_losses_71341

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@??o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:?????????@??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
]
A__inference_gsc_13_layer_call_and_return_conditional_losses_71711

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_gsc_6_layer_call_and_return_conditional_losses_69024

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ppm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
?
'__inference_merge_1_layer_call_fn_71752

inputs%
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_1_layer_call_and_return_conditional_losses_69192{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:????????? 
 
_user_specified_nameinputs
?
B
&__inference_fuse_2_layer_call_fn_71738

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_fuse_2_layer_call_and_return_conditional_losses_68826?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
\
@__inference_opt_4_layer_call_and_return_conditional_losses_68742

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_gsc_10_layer_call_and_return_conditional_losses_71611

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_opt_10_layer_call_and_return_conditional_losses_71621

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_fuse_2_layer_call_and_return_conditional_losses_68826

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_opt_6_layer_call_and_return_conditional_losses_71501

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ppm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
[
?__inference_flat_layer_call_and_return_conditional_losses_71904

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
\
@__inference_opt_1_layer_call_and_return_conditional_losses_68880

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*
ellipsis_mask*
end_maskl
IdentityIdentitystrided_slice:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
R
&__inference_fuse_1_layer_call_fn_71727
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_fuse_1_layer_call_and_return_conditional_losses_69178l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????@ :?????????@ :] Y
3
_output_shapes!
:?????????@ 
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:?????????@ 
"
_user_specified_name
inputs/1
?
\
@__inference_opt_1_layer_call_and_return_conditional_losses_69777

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*
ellipsis_mask*
end_maskl
IdentityIdentitystrided_slice:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
@__inference_opt_9_layer_call_and_return_conditional_losses_69077

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 b
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 f
IdentityIdentitySigmoid:y:0^NoOp*
T0*3
_output_shapes!
:?????????@88 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@88 
 
_user_specified_nameinputs
?
[
?__inference_flat_layer_call_and_return_conditional_losses_69292

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
?
@__inference_gsc_5_layer_call_and_return_conditional_losses_71441

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ppm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
C
'__inference_merge_6_layer_call_fn_71838

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_6_layer_call_and_return_conditional_losses_68850?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
]
$__inference_fc_2_layer_call_fn_71934

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_2_layer_call_and_return_conditional_losses_69496p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
Ƞ
?+
 __inference__wrapped_model_68721
input_1K
-cnn_base_opt_2_conv3d_readvariableop_resource:<
.cnn_base_opt_2_biasadd_readvariableop_resource:K
-cnn_base_gsc_2_conv3d_readvariableop_resource:<
.cnn_base_gsc_2_biasadd_readvariableop_resource:K
-cnn_base_opt_3_conv3d_readvariableop_resource:<
.cnn_base_opt_3_biasadd_readvariableop_resource:K
-cnn_base_gsc_3_conv3d_readvariableop_resource:<
.cnn_base_gsc_3_biasadd_readvariableop_resource:K
-cnn_base_opt_5_conv3d_readvariableop_resource:<
.cnn_base_opt_5_biasadd_readvariableop_resource:K
-cnn_base_gsc_5_conv3d_readvariableop_resource:<
.cnn_base_gsc_5_biasadd_readvariableop_resource:K
-cnn_base_opt_6_conv3d_readvariableop_resource:<
.cnn_base_opt_6_biasadd_readvariableop_resource:K
-cnn_base_gsc_6_conv3d_readvariableop_resource:<
.cnn_base_gsc_6_biasadd_readvariableop_resource:K
-cnn_base_opt_8_conv3d_readvariableop_resource: <
.cnn_base_opt_8_biasadd_readvariableop_resource: K
-cnn_base_gsc_8_conv3d_readvariableop_resource: <
.cnn_base_gsc_8_biasadd_readvariableop_resource: K
-cnn_base_opt_9_conv3d_readvariableop_resource:  <
.cnn_base_opt_9_biasadd_readvariableop_resource: K
-cnn_base_gsc_9_conv3d_readvariableop_resource:  <
.cnn_base_gsc_9_biasadd_readvariableop_resource: L
.cnn_base_opt_11_conv3d_readvariableop_resource:  =
/cnn_base_opt_11_biasadd_readvariableop_resource: L
.cnn_base_gsc_11_conv3d_readvariableop_resource:  =
/cnn_base_gsc_11_biasadd_readvariableop_resource: L
.cnn_base_opt_12_conv3d_readvariableop_resource:  =
/cnn_base_opt_12_biasadd_readvariableop_resource: L
.cnn_base_gsc_12_conv3d_readvariableop_resource:  =
/cnn_base_gsc_12_biasadd_readvariableop_resource: M
/cnn_base_merge_1_conv3d_readvariableop_resource: @>
0cnn_base_merge_1_biasadd_readvariableop_resource:@M
/cnn_base_merge_2_conv3d_readvariableop_resource:@@>
0cnn_base_merge_2_biasadd_readvariableop_resource:@M
/cnn_base_merge_4_conv3d_readvariableop_resource:@@>
0cnn_base_merge_4_biasadd_readvariableop_resource:@M
/cnn_base_merge_5_conv3d_readvariableop_resource:@@>
0cnn_base_merge_5_biasadd_readvariableop_resource:@N
/cnn_base_merge_7_conv3d_readvariableop_resource:@??
0cnn_base_merge_7_biasadd_readvariableop_resource:	?O
/cnn_base_merge_8_conv3d_readvariableop_resource:???
0cnn_base_merge_8_biasadd_readvariableop_resource:	?@
,cnn_base_fc_1_matmul_readvariableop_resource:
??<
-cnn_base_fc_1_biasadd_readvariableop_resource:	??
,cnn_base_fc_3_matmul_readvariableop_resource:	? ;
-cnn_base_fc_3_biasadd_readvariableop_resource: >
,cnn_base_pred_matmul_readvariableop_resource: ;
-cnn_base_pred_biasadd_readvariableop_resource:
identity??$cnn_base/fc_1/BiasAdd/ReadVariableOp?#cnn_base/fc_1/MatMul/ReadVariableOp?$cnn_base/fc_3/BiasAdd/ReadVariableOp?#cnn_base/fc_3/MatMul/ReadVariableOp?&cnn_base/gsc_11/BiasAdd/ReadVariableOp?%cnn_base/gsc_11/Conv3D/ReadVariableOp?&cnn_base/gsc_12/BiasAdd/ReadVariableOp?%cnn_base/gsc_12/Conv3D/ReadVariableOp?%cnn_base/gsc_2/BiasAdd/ReadVariableOp?$cnn_base/gsc_2/Conv3D/ReadVariableOp?%cnn_base/gsc_3/BiasAdd/ReadVariableOp?$cnn_base/gsc_3/Conv3D/ReadVariableOp?%cnn_base/gsc_5/BiasAdd/ReadVariableOp?$cnn_base/gsc_5/Conv3D/ReadVariableOp?%cnn_base/gsc_6/BiasAdd/ReadVariableOp?$cnn_base/gsc_6/Conv3D/ReadVariableOp?%cnn_base/gsc_8/BiasAdd/ReadVariableOp?$cnn_base/gsc_8/Conv3D/ReadVariableOp?%cnn_base/gsc_9/BiasAdd/ReadVariableOp?$cnn_base/gsc_9/Conv3D/ReadVariableOp?'cnn_base/merge_1/BiasAdd/ReadVariableOp?&cnn_base/merge_1/Conv3D/ReadVariableOp?'cnn_base/merge_2/BiasAdd/ReadVariableOp?&cnn_base/merge_2/Conv3D/ReadVariableOp?'cnn_base/merge_4/BiasAdd/ReadVariableOp?&cnn_base/merge_4/Conv3D/ReadVariableOp?'cnn_base/merge_5/BiasAdd/ReadVariableOp?&cnn_base/merge_5/Conv3D/ReadVariableOp?'cnn_base/merge_7/BiasAdd/ReadVariableOp?&cnn_base/merge_7/Conv3D/ReadVariableOp?'cnn_base/merge_8/BiasAdd/ReadVariableOp?&cnn_base/merge_8/Conv3D/ReadVariableOp?&cnn_base/opt_11/BiasAdd/ReadVariableOp?%cnn_base/opt_11/Conv3D/ReadVariableOp?&cnn_base/opt_12/BiasAdd/ReadVariableOp?%cnn_base/opt_12/Conv3D/ReadVariableOp?%cnn_base/opt_2/BiasAdd/ReadVariableOp?$cnn_base/opt_2/Conv3D/ReadVariableOp?%cnn_base/opt_3/BiasAdd/ReadVariableOp?$cnn_base/opt_3/Conv3D/ReadVariableOp?%cnn_base/opt_5/BiasAdd/ReadVariableOp?$cnn_base/opt_5/Conv3D/ReadVariableOp?%cnn_base/opt_6/BiasAdd/ReadVariableOp?$cnn_base/opt_6/Conv3D/ReadVariableOp?%cnn_base/opt_8/BiasAdd/ReadVariableOp?$cnn_base/opt_8/Conv3D/ReadVariableOp?%cnn_base/opt_9/BiasAdd/ReadVariableOp?$cnn_base/opt_9/Conv3D/ReadVariableOp?$cnn_base/pred/BiasAdd/ReadVariableOp?#cnn_base/pred/MatMul/ReadVariableOps
"cnn_base/opt_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$cnn_base/opt_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$cnn_base/opt_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
cnn_base/opt_1/strided_sliceStridedSliceinput_1+cnn_base/opt_1/strided_slice/stack:output:0-cnn_base/opt_1/strided_slice/stack_1:output:0-cnn_base/opt_1/strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*
ellipsis_mask*
end_masks
"cnn_base/gsc_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$cnn_base/gsc_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$cnn_base/gsc_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
cnn_base/gsc_1/strided_sliceStridedSliceinput_1+cnn_base/gsc_1/strided_slice/stack:output:0-cnn_base/gsc_1/strided_slice/stack_1:output:0-cnn_base/gsc_1/strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*

begin_mask*
ellipsis_mask?
$cnn_base/opt_2/Conv3D/ReadVariableOpReadVariableOp-cnn_base_opt_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
cnn_base/opt_2/Conv3DConv3D%cnn_base/opt_1/strided_slice:output:0,cnn_base/opt_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
?
%cnn_base/opt_2/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_opt_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
cnn_base/opt_2/BiasAddBiasAddcnn_base/opt_2/Conv3D:output:0-cnn_base/opt_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??|
cnn_base/opt_2/ReluRelucnn_base/opt_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
$cnn_base/gsc_2/Conv3D/ReadVariableOpReadVariableOp-cnn_base_gsc_2_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
cnn_base/gsc_2/Conv3DConv3D%cnn_base/gsc_1/strided_slice:output:0,cnn_base/gsc_2/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
?
%cnn_base/gsc_2/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_gsc_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
cnn_base/gsc_2/BiasAddBiasAddcnn_base/gsc_2/Conv3D:output:0-cnn_base/gsc_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??|
cnn_base/gsc_2/ReluRelucnn_base/gsc_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
$cnn_base/opt_3/Conv3D/ReadVariableOpReadVariableOp-cnn_base_opt_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
cnn_base/opt_3/Conv3DConv3D!cnn_base/opt_2/Relu:activations:0,cnn_base/opt_3/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
?
%cnn_base/opt_3/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_opt_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
cnn_base/opt_3/BiasAddBiasAddcnn_base/opt_3/Conv3D:output:0-cnn_base/opt_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??|
cnn_base/opt_3/ReluRelucnn_base/opt_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
$cnn_base/gsc_3/Conv3D/ReadVariableOpReadVariableOp-cnn_base_gsc_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
cnn_base/gsc_3/Conv3DConv3D!cnn_base/gsc_2/Relu:activations:0,cnn_base/gsc_3/Conv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
?
%cnn_base/gsc_3/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_gsc_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
cnn_base/gsc_3/BiasAddBiasAddcnn_base/gsc_3/Conv3D:output:0-cnn_base/gsc_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??|
cnn_base/gsc_3/ReluRelucnn_base/gsc_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@???
cnn_base/opt_4/MaxPool3D	MaxPool3D!cnn_base/opt_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????@pp*
ksize	
*
paddingVALID*
strides	
?
cnn_base/gsc_4/MaxPool3D	MaxPool3D!cnn_base/gsc_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????@pp*
ksize	
*
paddingVALID*
strides	
?
$cnn_base/opt_5/Conv3D/ReadVariableOpReadVariableOp-cnn_base_opt_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
cnn_base/opt_5/Conv3DConv3D!cnn_base/opt_4/MaxPool3D:output:0,cnn_base/opt_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
?
%cnn_base/opt_5/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_opt_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
cnn_base/opt_5/BiasAddBiasAddcnn_base/opt_5/Conv3D:output:0-cnn_base/opt_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ppz
cnn_base/opt_5/ReluRelucnn_base/opt_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
$cnn_base/gsc_5/Conv3D/ReadVariableOpReadVariableOp-cnn_base_gsc_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
cnn_base/gsc_5/Conv3DConv3D!cnn_base/gsc_4/MaxPool3D:output:0,cnn_base/gsc_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
?
%cnn_base/gsc_5/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_gsc_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
cnn_base/gsc_5/BiasAddBiasAddcnn_base/gsc_5/Conv3D:output:0-cnn_base/gsc_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ppz
cnn_base/gsc_5/ReluRelucnn_base/gsc_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
$cnn_base/opt_6/Conv3D/ReadVariableOpReadVariableOp-cnn_base_opt_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
cnn_base/opt_6/Conv3DConv3D!cnn_base/opt_5/Relu:activations:0,cnn_base/opt_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
?
%cnn_base/opt_6/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_opt_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
cnn_base/opt_6/BiasAddBiasAddcnn_base/opt_6/Conv3D:output:0-cnn_base/opt_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ppz
cnn_base/opt_6/ReluRelucnn_base/opt_6/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
$cnn_base/gsc_6/Conv3D/ReadVariableOpReadVariableOp-cnn_base_gsc_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
cnn_base/gsc_6/Conv3DConv3D!cnn_base/gsc_5/Relu:activations:0,cnn_base/gsc_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
?
%cnn_base/gsc_6/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_gsc_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
cnn_base/gsc_6/BiasAddBiasAddcnn_base/gsc_6/Conv3D:output:0-cnn_base/gsc_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ppz
cnn_base/gsc_6/ReluRelucnn_base/gsc_6/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@pp?
cnn_base/opt_7/MaxPool3D	MaxPool3D!cnn_base/opt_6/Relu:activations:0*
T0*3
_output_shapes!
:?????????@88*
ksize	
*
paddingVALID*
strides	
?
cnn_base/gsc_7/MaxPool3D	MaxPool3D!cnn_base/gsc_6/Relu:activations:0*
T0*3
_output_shapes!
:?????????@88*
ksize	
*
paddingVALID*
strides	
?
$cnn_base/opt_8/Conv3D/ReadVariableOpReadVariableOp-cnn_base_opt_8_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0?
cnn_base/opt_8/Conv3DConv3D!cnn_base/opt_7/MaxPool3D:output:0,cnn_base/opt_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
?
%cnn_base/opt_8/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_opt_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_base/opt_8/BiasAddBiasAddcnn_base/opt_8/Conv3D:output:0-cnn_base/opt_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 ?
cnn_base/opt_8/SigmoidSigmoidcnn_base/opt_8/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
$cnn_base/gsc_8/Conv3D/ReadVariableOpReadVariableOp-cnn_base_gsc_8_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0?
cnn_base/gsc_8/Conv3DConv3D!cnn_base/gsc_7/MaxPool3D:output:0,cnn_base/gsc_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
?
%cnn_base/gsc_8/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_gsc_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_base/gsc_8/BiasAddBiasAddcnn_base/gsc_8/Conv3D:output:0-cnn_base/gsc_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 z
cnn_base/gsc_8/ReluRelucnn_base/gsc_8/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
$cnn_base/opt_9/Conv3D/ReadVariableOpReadVariableOp-cnn_base_opt_9_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
cnn_base/opt_9/Conv3DConv3Dcnn_base/opt_8/Sigmoid:y:0,cnn_base/opt_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
?
%cnn_base/opt_9/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_opt_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_base/opt_9/BiasAddBiasAddcnn_base/opt_9/Conv3D:output:0-cnn_base/opt_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 ?
cnn_base/opt_9/SigmoidSigmoidcnn_base/opt_9/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
$cnn_base/gsc_9/Conv3D/ReadVariableOpReadVariableOp-cnn_base_gsc_9_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
cnn_base/gsc_9/Conv3DConv3D!cnn_base/gsc_8/Relu:activations:0,cnn_base/gsc_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
?
%cnn_base/gsc_9/BiasAdd/ReadVariableOpReadVariableOp.cnn_base_gsc_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_base/gsc_9/BiasAddBiasAddcnn_base/gsc_9/Conv3D:output:0-cnn_base/gsc_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 z
cnn_base/gsc_9/ReluRelucnn_base/gsc_9/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 ?
cnn_base/opt_10/MaxPool3D	MaxPool3Dcnn_base/opt_9/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?
cnn_base/gsc_10/MaxPool3D	MaxPool3D!cnn_base/gsc_9/Relu:activations:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?
%cnn_base/opt_11/Conv3D/ReadVariableOpReadVariableOp.cnn_base_opt_11_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
cnn_base/opt_11/Conv3DConv3D"cnn_base/opt_10/MaxPool3D:output:0-cnn_base/opt_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
&cnn_base/opt_11/BiasAdd/ReadVariableOpReadVariableOp/cnn_base_opt_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_base/opt_11/BiasAddBiasAddcnn_base/opt_11/Conv3D:output:0.cnn_base/opt_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ ?
cnn_base/opt_11/SigmoidSigmoid cnn_base/opt_11/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
%cnn_base/gsc_11/Conv3D/ReadVariableOpReadVariableOp.cnn_base_gsc_11_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
cnn_base/gsc_11/Conv3DConv3D"cnn_base/gsc_10/MaxPool3D:output:0-cnn_base/gsc_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
&cnn_base/gsc_11/BiasAdd/ReadVariableOpReadVariableOp/cnn_base_gsc_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_base/gsc_11/BiasAddBiasAddcnn_base/gsc_11/Conv3D:output:0.cnn_base/gsc_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ |
cnn_base/gsc_11/ReluRelu cnn_base/gsc_11/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
%cnn_base/opt_12/Conv3D/ReadVariableOpReadVariableOp.cnn_base_opt_12_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
cnn_base/opt_12/Conv3DConv3Dcnn_base/opt_11/Sigmoid:y:0-cnn_base/opt_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
&cnn_base/opt_12/BiasAdd/ReadVariableOpReadVariableOp/cnn_base_opt_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_base/opt_12/BiasAddBiasAddcnn_base/opt_12/Conv3D:output:0.cnn_base/opt_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ ?
cnn_base/opt_12/SigmoidSigmoid cnn_base/opt_12/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
%cnn_base/gsc_12/Conv3D/ReadVariableOpReadVariableOp.cnn_base_gsc_12_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
cnn_base/gsc_12/Conv3DConv3D"cnn_base/gsc_11/Relu:activations:0-cnn_base/gsc_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
?
&cnn_base/gsc_12/BiasAdd/ReadVariableOpReadVariableOp/cnn_base_gsc_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_base/gsc_12/BiasAddBiasAddcnn_base/gsc_12/Conv3D:output:0.cnn_base/gsc_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ |
cnn_base/gsc_12/ReluRelu cnn_base/gsc_12/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ ?
cnn_base/gsc_13/MaxPool3D	MaxPool3D"cnn_base/gsc_12/Relu:activations:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?
cnn_base/opt_13/MaxPool3D	MaxPool3Dcnn_base/opt_12/Sigmoid:y:0*
T0*3
_output_shapes!
:?????????@ *
ksize	
*
paddingVALID*
strides	
?
cnn_base/fuse_1/mulMul"cnn_base/gsc_13/MaxPool3D:output:0"cnn_base/opt_13/MaxPool3D:output:0*
T0*3
_output_shapes!
:?????????@ ?
cnn_base/fuse_2/MaxPool3D	MaxPool3Dcnn_base/fuse_1/mul:z:0*
T0*3
_output_shapes!
:????????? *
ksize	
*
paddingVALID*
strides	
?
&cnn_base/merge_1/Conv3D/ReadVariableOpReadVariableOp/cnn_base_merge_1_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0?
cnn_base/merge_1/Conv3DConv3D"cnn_base/fuse_2/MaxPool3D:output:0.cnn_base/merge_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
'cnn_base/merge_1/BiasAdd/ReadVariableOpReadVariableOp0cnn_base_merge_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
cnn_base/merge_1/BiasAddBiasAdd cnn_base/merge_1/Conv3D:output:0/cnn_base/merge_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@~
cnn_base/merge_1/ReluRelu!cnn_base/merge_1/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
&cnn_base/merge_2/Conv3D/ReadVariableOpReadVariableOp/cnn_base_merge_2_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
cnn_base/merge_2/Conv3DConv3D#cnn_base/merge_1/Relu:activations:0.cnn_base/merge_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
'cnn_base/merge_2/BiasAdd/ReadVariableOpReadVariableOp0cnn_base_merge_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
cnn_base/merge_2/BiasAddBiasAdd cnn_base/merge_2/Conv3D:output:0/cnn_base/merge_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@~
cnn_base/merge_2/ReluRelu!cnn_base/merge_2/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
cnn_base/merge_3/MaxPool3D	MaxPool3D#cnn_base/merge_2/Relu:activations:0*
T0*3
_output_shapes!
:?????????@*
ksize	
*
paddingVALID*
strides	
?
&cnn_base/merge_4/Conv3D/ReadVariableOpReadVariableOp/cnn_base_merge_4_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
cnn_base/merge_4/Conv3DConv3D#cnn_base/merge_3/MaxPool3D:output:0.cnn_base/merge_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
'cnn_base/merge_4/BiasAdd/ReadVariableOpReadVariableOp0cnn_base_merge_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
cnn_base/merge_4/BiasAddBiasAdd cnn_base/merge_4/Conv3D:output:0/cnn_base/merge_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@~
cnn_base/merge_4/ReluRelu!cnn_base/merge_4/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
&cnn_base/merge_5/Conv3D/ReadVariableOpReadVariableOp/cnn_base_merge_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
cnn_base/merge_5/Conv3DConv3D#cnn_base/merge_4/Relu:activations:0.cnn_base/merge_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
?
'cnn_base/merge_5/BiasAdd/ReadVariableOpReadVariableOp0cnn_base_merge_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
cnn_base/merge_5/BiasAddBiasAdd cnn_base/merge_5/Conv3D:output:0/cnn_base/merge_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@~
cnn_base/merge_5/ReluRelu!cnn_base/merge_5/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????@?
cnn_base/merge_6/MaxPool3D	MaxPool3D#cnn_base/merge_5/Relu:activations:0*
T0*3
_output_shapes!
:?????????@*
ksize	
*
paddingVALID*
strides	
?
&cnn_base/merge_7/Conv3D/ReadVariableOpReadVariableOp/cnn_base_merge_7_conv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype0?
cnn_base/merge_7/Conv3DConv3D#cnn_base/merge_6/MaxPool3D:output:0.cnn_base/merge_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
?
'cnn_base/merge_7/BiasAdd/ReadVariableOpReadVariableOp0cnn_base_merge_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
cnn_base/merge_7/BiasAddBiasAdd cnn_base/merge_7/Conv3D:output:0/cnn_base/merge_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????
cnn_base/merge_7/ReluRelu!cnn_base/merge_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :???????????
&cnn_base/merge_8/Conv3D/ReadVariableOpReadVariableOp/cnn_base_merge_8_conv3d_readvariableop_resource*,
_output_shapes
:??*
dtype0?
cnn_base/merge_8/Conv3DConv3D#cnn_base/merge_7/Relu:activations:0.cnn_base/merge_8/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
?
'cnn_base/merge_8/BiasAdd/ReadVariableOpReadVariableOp0cnn_base_merge_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
cnn_base/merge_8/BiasAddBiasAdd cnn_base/merge_8/Conv3D:output:0/cnn_base/merge_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????
cnn_base/merge_8/ReluRelu!cnn_base/merge_8/BiasAdd:output:0*
T0*4
_output_shapes"
 :???????????
cnn_base/merge_9/MaxPool3D	MaxPool3D#cnn_base/merge_8/Relu:activations:0*
T0*4
_output_shapes"
 :??????????*
ksize	
*
paddingVALID*
strides	
d
cnn_base/flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
cnn_base/flat/ReshapeReshape#cnn_base/merge_9/MaxPool3D:output:0cnn_base/flat/Const:output:0*
T0*(
_output_shapes
:???????????
#cnn_base/fc_1/MatMul/ReadVariableOpReadVariableOp,cnn_base_fc_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
cnn_base/fc_1/MatMulMatMulcnn_base/flat/Reshape:output:0+cnn_base/fc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$cnn_base/fc_1/BiasAdd/ReadVariableOpReadVariableOp-cnn_base_fc_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
cnn_base/fc_1/BiasAddBiasAddcnn_base/fc_1/MatMul:product:0,cnn_base/fc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????m
cnn_base/fc_1/ReluRelucnn_base/fc_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????w
cnn_base/fc_2/IdentityIdentity cnn_base/fc_1/Relu:activations:0*
T0*(
_output_shapes
:???????????
#cnn_base/fc_3/MatMul/ReadVariableOpReadVariableOp,cnn_base_fc_3_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
cnn_base/fc_3/MatMulMatMulcnn_base/fc_2/Identity:output:0+cnn_base/fc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
$cnn_base/fc_3/BiasAdd/ReadVariableOpReadVariableOp-cnn_base_fc_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_base/fc_3/BiasAddBiasAddcnn_base/fc_3/MatMul:product:0,cnn_base/fc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? l
cnn_base/fc_3/ReluRelucnn_base/fc_3/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
#cnn_base/pred/MatMul/ReadVariableOpReadVariableOp,cnn_base_pred_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
cnn_base/pred/MatMulMatMul cnn_base/fc_3/Relu:activations:0+cnn_base/pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$cnn_base/pred/BiasAdd/ReadVariableOpReadVariableOp-cnn_base_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
cnn_base/pred/BiasAddBiasAddcnn_base/pred/MatMul:product:0,cnn_base/pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
cnn_base/pred/SigmoidSigmoidcnn_base/pred/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitycnn_base/pred/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp%^cnn_base/fc_1/BiasAdd/ReadVariableOp$^cnn_base/fc_1/MatMul/ReadVariableOp%^cnn_base/fc_3/BiasAdd/ReadVariableOp$^cnn_base/fc_3/MatMul/ReadVariableOp'^cnn_base/gsc_11/BiasAdd/ReadVariableOp&^cnn_base/gsc_11/Conv3D/ReadVariableOp'^cnn_base/gsc_12/BiasAdd/ReadVariableOp&^cnn_base/gsc_12/Conv3D/ReadVariableOp&^cnn_base/gsc_2/BiasAdd/ReadVariableOp%^cnn_base/gsc_2/Conv3D/ReadVariableOp&^cnn_base/gsc_3/BiasAdd/ReadVariableOp%^cnn_base/gsc_3/Conv3D/ReadVariableOp&^cnn_base/gsc_5/BiasAdd/ReadVariableOp%^cnn_base/gsc_5/Conv3D/ReadVariableOp&^cnn_base/gsc_6/BiasAdd/ReadVariableOp%^cnn_base/gsc_6/Conv3D/ReadVariableOp&^cnn_base/gsc_8/BiasAdd/ReadVariableOp%^cnn_base/gsc_8/Conv3D/ReadVariableOp&^cnn_base/gsc_9/BiasAdd/ReadVariableOp%^cnn_base/gsc_9/Conv3D/ReadVariableOp(^cnn_base/merge_1/BiasAdd/ReadVariableOp'^cnn_base/merge_1/Conv3D/ReadVariableOp(^cnn_base/merge_2/BiasAdd/ReadVariableOp'^cnn_base/merge_2/Conv3D/ReadVariableOp(^cnn_base/merge_4/BiasAdd/ReadVariableOp'^cnn_base/merge_4/Conv3D/ReadVariableOp(^cnn_base/merge_5/BiasAdd/ReadVariableOp'^cnn_base/merge_5/Conv3D/ReadVariableOp(^cnn_base/merge_7/BiasAdd/ReadVariableOp'^cnn_base/merge_7/Conv3D/ReadVariableOp(^cnn_base/merge_8/BiasAdd/ReadVariableOp'^cnn_base/merge_8/Conv3D/ReadVariableOp'^cnn_base/opt_11/BiasAdd/ReadVariableOp&^cnn_base/opt_11/Conv3D/ReadVariableOp'^cnn_base/opt_12/BiasAdd/ReadVariableOp&^cnn_base/opt_12/Conv3D/ReadVariableOp&^cnn_base/opt_2/BiasAdd/ReadVariableOp%^cnn_base/opt_2/Conv3D/ReadVariableOp&^cnn_base/opt_3/BiasAdd/ReadVariableOp%^cnn_base/opt_3/Conv3D/ReadVariableOp&^cnn_base/opt_5/BiasAdd/ReadVariableOp%^cnn_base/opt_5/Conv3D/ReadVariableOp&^cnn_base/opt_6/BiasAdd/ReadVariableOp%^cnn_base/opt_6/Conv3D/ReadVariableOp&^cnn_base/opt_8/BiasAdd/ReadVariableOp%^cnn_base/opt_8/Conv3D/ReadVariableOp&^cnn_base/opt_9/BiasAdd/ReadVariableOp%^cnn_base/opt_9/Conv3D/ReadVariableOp%^cnn_base/pred/BiasAdd/ReadVariableOp$^cnn_base/pred/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$cnn_base/fc_1/BiasAdd/ReadVariableOp$cnn_base/fc_1/BiasAdd/ReadVariableOp2J
#cnn_base/fc_1/MatMul/ReadVariableOp#cnn_base/fc_1/MatMul/ReadVariableOp2L
$cnn_base/fc_3/BiasAdd/ReadVariableOp$cnn_base/fc_3/BiasAdd/ReadVariableOp2J
#cnn_base/fc_3/MatMul/ReadVariableOp#cnn_base/fc_3/MatMul/ReadVariableOp2P
&cnn_base/gsc_11/BiasAdd/ReadVariableOp&cnn_base/gsc_11/BiasAdd/ReadVariableOp2N
%cnn_base/gsc_11/Conv3D/ReadVariableOp%cnn_base/gsc_11/Conv3D/ReadVariableOp2P
&cnn_base/gsc_12/BiasAdd/ReadVariableOp&cnn_base/gsc_12/BiasAdd/ReadVariableOp2N
%cnn_base/gsc_12/Conv3D/ReadVariableOp%cnn_base/gsc_12/Conv3D/ReadVariableOp2N
%cnn_base/gsc_2/BiasAdd/ReadVariableOp%cnn_base/gsc_2/BiasAdd/ReadVariableOp2L
$cnn_base/gsc_2/Conv3D/ReadVariableOp$cnn_base/gsc_2/Conv3D/ReadVariableOp2N
%cnn_base/gsc_3/BiasAdd/ReadVariableOp%cnn_base/gsc_3/BiasAdd/ReadVariableOp2L
$cnn_base/gsc_3/Conv3D/ReadVariableOp$cnn_base/gsc_3/Conv3D/ReadVariableOp2N
%cnn_base/gsc_5/BiasAdd/ReadVariableOp%cnn_base/gsc_5/BiasAdd/ReadVariableOp2L
$cnn_base/gsc_5/Conv3D/ReadVariableOp$cnn_base/gsc_5/Conv3D/ReadVariableOp2N
%cnn_base/gsc_6/BiasAdd/ReadVariableOp%cnn_base/gsc_6/BiasAdd/ReadVariableOp2L
$cnn_base/gsc_6/Conv3D/ReadVariableOp$cnn_base/gsc_6/Conv3D/ReadVariableOp2N
%cnn_base/gsc_8/BiasAdd/ReadVariableOp%cnn_base/gsc_8/BiasAdd/ReadVariableOp2L
$cnn_base/gsc_8/Conv3D/ReadVariableOp$cnn_base/gsc_8/Conv3D/ReadVariableOp2N
%cnn_base/gsc_9/BiasAdd/ReadVariableOp%cnn_base/gsc_9/BiasAdd/ReadVariableOp2L
$cnn_base/gsc_9/Conv3D/ReadVariableOp$cnn_base/gsc_9/Conv3D/ReadVariableOp2R
'cnn_base/merge_1/BiasAdd/ReadVariableOp'cnn_base/merge_1/BiasAdd/ReadVariableOp2P
&cnn_base/merge_1/Conv3D/ReadVariableOp&cnn_base/merge_1/Conv3D/ReadVariableOp2R
'cnn_base/merge_2/BiasAdd/ReadVariableOp'cnn_base/merge_2/BiasAdd/ReadVariableOp2P
&cnn_base/merge_2/Conv3D/ReadVariableOp&cnn_base/merge_2/Conv3D/ReadVariableOp2R
'cnn_base/merge_4/BiasAdd/ReadVariableOp'cnn_base/merge_4/BiasAdd/ReadVariableOp2P
&cnn_base/merge_4/Conv3D/ReadVariableOp&cnn_base/merge_4/Conv3D/ReadVariableOp2R
'cnn_base/merge_5/BiasAdd/ReadVariableOp'cnn_base/merge_5/BiasAdd/ReadVariableOp2P
&cnn_base/merge_5/Conv3D/ReadVariableOp&cnn_base/merge_5/Conv3D/ReadVariableOp2R
'cnn_base/merge_7/BiasAdd/ReadVariableOp'cnn_base/merge_7/BiasAdd/ReadVariableOp2P
&cnn_base/merge_7/Conv3D/ReadVariableOp&cnn_base/merge_7/Conv3D/ReadVariableOp2R
'cnn_base/merge_8/BiasAdd/ReadVariableOp'cnn_base/merge_8/BiasAdd/ReadVariableOp2P
&cnn_base/merge_8/Conv3D/ReadVariableOp&cnn_base/merge_8/Conv3D/ReadVariableOp2P
&cnn_base/opt_11/BiasAdd/ReadVariableOp&cnn_base/opt_11/BiasAdd/ReadVariableOp2N
%cnn_base/opt_11/Conv3D/ReadVariableOp%cnn_base/opt_11/Conv3D/ReadVariableOp2P
&cnn_base/opt_12/BiasAdd/ReadVariableOp&cnn_base/opt_12/BiasAdd/ReadVariableOp2N
%cnn_base/opt_12/Conv3D/ReadVariableOp%cnn_base/opt_12/Conv3D/ReadVariableOp2N
%cnn_base/opt_2/BiasAdd/ReadVariableOp%cnn_base/opt_2/BiasAdd/ReadVariableOp2L
$cnn_base/opt_2/Conv3D/ReadVariableOp$cnn_base/opt_2/Conv3D/ReadVariableOp2N
%cnn_base/opt_3/BiasAdd/ReadVariableOp%cnn_base/opt_3/BiasAdd/ReadVariableOp2L
$cnn_base/opt_3/Conv3D/ReadVariableOp$cnn_base/opt_3/Conv3D/ReadVariableOp2N
%cnn_base/opt_5/BiasAdd/ReadVariableOp%cnn_base/opt_5/BiasAdd/ReadVariableOp2L
$cnn_base/opt_5/Conv3D/ReadVariableOp$cnn_base/opt_5/Conv3D/ReadVariableOp2N
%cnn_base/opt_6/BiasAdd/ReadVariableOp%cnn_base/opt_6/BiasAdd/ReadVariableOp2L
$cnn_base/opt_6/Conv3D/ReadVariableOp$cnn_base/opt_6/Conv3D/ReadVariableOp2N
%cnn_base/opt_8/BiasAdd/ReadVariableOp%cnn_base/opt_8/BiasAdd/ReadVariableOp2L
$cnn_base/opt_8/Conv3D/ReadVariableOp$cnn_base/opt_8/Conv3D/ReadVariableOp2N
%cnn_base/opt_9/BiasAdd/ReadVariableOp%cnn_base/opt_9/BiasAdd/ReadVariableOp2L
$cnn_base/opt_9/Conv3D/ReadVariableOp$cnn_base/opt_9/Conv3D/ReadVariableOp2L
$cnn_base/pred/BiasAdd/ReadVariableOp$cnn_base/pred/BiasAdd/ReadVariableOp2J
#cnn_base/pred/MatMul/ReadVariableOp#cnn_base/pred/MatMul/ReadVariableOp:^ Z
5
_output_shapes#
!:?????????@??
!
_user_specified_name	input_1
?
?
@__inference_opt_5_layer_call_and_return_conditional_losses_68973

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ppm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
\
@__inference_opt_7_layer_call_and_return_conditional_losses_68766

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_opt_10_layer_call_and_return_conditional_losses_68790

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_merge_2_layer_call_and_return_conditional_losses_69209

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?

?
?__inference_fc_1_layer_call_and_return_conditional_losses_71924

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_opt_11_layer_call_fn_71650

inputs%
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_11_layer_call_and_return_conditional_losses_69113{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
?
?
B__inference_merge_8_layer_call_and_return_conditional_losses_71883

inputs>
conv3d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:??*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
?
&__inference_gsc_11_layer_call_fn_71630

inputs%
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_11_layer_call_and_return_conditional_losses_69130{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
?
?
A__inference_opt_11_layer_call_and_return_conditional_losses_69113

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ b
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ f
IdentityIdentitySigmoid:y:0^NoOp*
T0*3
_output_shapes!
:?????????@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
?
?
B__inference_merge_7_layer_call_and_return_conditional_losses_69262

inputs=
conv3d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_merge_8_layer_call_and_return_conditional_losses_69279

inputs>
conv3d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:??*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
^
B__inference_merge_9_layer_call_and_return_conditional_losses_71893

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_gsc_2_layer_call_and_return_conditional_losses_68920

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@??o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:?????????@??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
@__inference_gsc_9_layer_call_and_return_conditional_losses_69094

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@88 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@88 
 
_user_specified_nameinputs
?
^
B__inference_merge_6_layer_call_and_return_conditional_losses_68850

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_gsc_8_layer_call_and_return_conditional_losses_71541

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@88 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@88
 
_user_specified_nameinputs
?
?
B__inference_merge_4_layer_call_and_return_conditional_losses_69227

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_cnn_base_layer_call_fn_70241
input_1%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15: 

unknown_16: (

unknown_17: 

unknown_18: (

unknown_19:  

unknown_20: (

unknown_21:  

unknown_22: (

unknown_23:  

unknown_24: (

unknown_25:  

unknown_26: (

unknown_27:  

unknown_28: (

unknown_29:  

unknown_30: (

unknown_31: @

unknown_32:@(

unknown_33:@@

unknown_34:@(

unknown_35:@@

unknown_36:@(

unknown_37:@@

unknown_38:@)

unknown_39:@?

unknown_40:	?*

unknown_41:??

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:	? 

unknown_46: 

unknown_47: 

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_cnn_base_layer_call_and_return_conditional_losses_70033o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:?????????@??
!
_user_specified_name	input_1
?
]
A__inference_gsc_10_layer_call_and_return_conditional_losses_68778

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_merge_9_layer_call_and_return_conditional_losses_68862

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
A
%__inference_gsc_1_layer_call_fn_71274

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_1_layer_call_and_return_conditional_losses_68890n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
\
@__inference_opt_4_layer_call_and_return_conditional_losses_71421

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_merge_6_layer_call_and_return_conditional_losses_71843

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_opt_8_layer_call_and_return_conditional_losses_69043

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 b
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 f
IdentityIdentitySigmoid:y:0^NoOp*
T0*3
_output_shapes!
:?????????@88 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@88
 
_user_specified_nameinputs
?
?
@__inference_gsc_6_layer_call_and_return_conditional_losses_71481

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@pp\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ppm
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@pp
 
_user_specified_nameinputs
?
m
A__inference_fuse_1_layer_call_and_return_conditional_losses_71733
inputs_0
inputs_1
identity\
mulMulinputs_0inputs_1*
T0*3
_output_shapes!
:?????????@ [
IdentityIdentitymul:z:0*
T0*3
_output_shapes!
:?????????@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????@ :?????????@ :] Y
3
_output_shapes!
:?????????@ 
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:?????????@ 
"
_user_specified_name
inputs/1
?
?
'__inference_merge_8_layer_call_fn_71872

inputs'
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_8_layer_call_and_return_conditional_losses_69279|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????
 
_user_specified_nameinputs
?
?
'__inference_merge_4_layer_call_fn_71802

inputs%
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_4_layer_call_and_return_conditional_losses_69227{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
?
&__inference_opt_12_layer_call_fn_71690

inputs%
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_12_layer_call_and_return_conditional_losses_69147{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
?
?
A__inference_gsc_12_layer_call_and_return_conditional_losses_69164

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
?
A
%__inference_gsc_7_layer_call_fn_71506

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_7_layer_call_and_return_conditional_losses_68754?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_gsc_12_layer_call_fn_71670

inputs%
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_12_layer_call_and_return_conditional_losses_69164{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:?????????@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
?
?
B__inference_merge_1_layer_call_and_return_conditional_losses_69192

inputs<
conv3d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:????????? 
 
_user_specified_nameinputs
?
?
'__inference_merge_7_layer_call_fn_71852

inputs&
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_7_layer_call_and_return_conditional_losses_69262|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
\
@__inference_gsc_1_layer_call_and_return_conditional_losses_68890

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*5
_output_shapes#
!:?????????@??*

begin_mask*
ellipsis_maskl
IdentityIdentitystrided_slice:output:0*
T0*5
_output_shapes#
!:?????????@??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
?
A__inference_opt_11_layer_call_and_return_conditional_losses_71661

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@ b
SigmoidSigmoidBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@ f
IdentityIdentitySigmoid:y:0^NoOp*
T0*3
_output_shapes!
:?????????@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@ 
 
_user_specified_nameinputs
?
?
B__inference_merge_7_layer_call_and_return_conditional_losses_71863

inputs=
conv3d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@?*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_merge_5_layer_call_and_return_conditional_losses_69244

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@
 
_user_specified_nameinputs
?
B
&__inference_gsc_10_layer_call_fn_71606

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_10_layer_call_and_return_conditional_losses_68778?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?	
^
?__inference_fc_2_layer_call_and_return_conditional_losses_71951

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
?__inference_pred_layer_call_and_return_conditional_losses_69346

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
(__inference_cnn_base_layer_call_fn_69456
input_1%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3:
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15: 

unknown_16: (

unknown_17: 

unknown_18: (

unknown_19:  

unknown_20: (

unknown_21:  

unknown_22: (

unknown_23:  

unknown_24: (

unknown_25:  

unknown_26: (

unknown_27:  

unknown_28: (

unknown_29:  

unknown_30: (

unknown_31: @

unknown_32:@(

unknown_33:@@

unknown_34:@(

unknown_35:@@

unknown_36:@(

unknown_37:@@

unknown_38:@)

unknown_39:@?

unknown_40:	?*

unknown_41:??

unknown_42:	?

unknown_43:
??

unknown_44:	?

unknown_45:	? 

unknown_46: 

unknown_47: 

unknown_48:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_cnn_base_layer_call_and_return_conditional_losses_69353o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
5
_output_shapes#
!:?????????@??
!
_user_specified_name	input_1
?
\
@__inference_gsc_7_layer_call_and_return_conditional_losses_71511

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_gsc_9_layer_call_and_return_conditional_losses_71581

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????@88 \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????@88 m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:?????????@88 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@88 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????@88 
 
_user_specified_nameinputs
?
?
@__inference_opt_2_layer_call_and_return_conditional_losses_71361

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:?????????@??^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:?????????@??o
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:?????????@??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:] Y
5
_output_shapes#
!:?????????@??
 
_user_specified_nameinputs
?
\
@__inference_gsc_7_layer_call_and_return_conditional_losses_68754

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
??
?
C__inference_cnn_base_layer_call_and_return_conditional_losses_70387
input_1)
opt_2_70246:
opt_2_70248:)
gsc_2_70251:
gsc_2_70253:)
opt_3_70256:
opt_3_70258:)
gsc_3_70261:
gsc_3_70263:)
opt_5_70268:
opt_5_70270:)
gsc_5_70273:
gsc_5_70275:)
opt_6_70278:
opt_6_70280:)
gsc_6_70283:
gsc_6_70285:)
opt_8_70290: 
opt_8_70292: )
gsc_8_70295: 
gsc_8_70297: )
opt_9_70300:  
opt_9_70302: )
gsc_9_70305:  
gsc_9_70307: *
opt_11_70312:  
opt_11_70314: *
gsc_11_70317:  
gsc_11_70319: *
opt_12_70322:  
opt_12_70324: *
gsc_12_70327:  
gsc_12_70329: +
merge_1_70336: @
merge_1_70338:@+
merge_2_70341:@@
merge_2_70343:@+
merge_4_70347:@@
merge_4_70349:@+
merge_5_70352:@@
merge_5_70354:@,
merge_7_70358:@?
merge_7_70360:	?-
merge_8_70363:??
merge_8_70365:	?

fc_1_70370:
??

fc_1_70372:	?

fc_3_70376:	? 

fc_3_70378: 

pred_70381: 

pred_70383:
identity??fc_1/StatefulPartitionedCall?fc_3/StatefulPartitionedCall?gsc_11/StatefulPartitionedCall?gsc_12/StatefulPartitionedCall?gsc_2/StatefulPartitionedCall?gsc_3/StatefulPartitionedCall?gsc_5/StatefulPartitionedCall?gsc_6/StatefulPartitionedCall?gsc_8/StatefulPartitionedCall?gsc_9/StatefulPartitionedCall?merge_1/StatefulPartitionedCall?merge_2/StatefulPartitionedCall?merge_4/StatefulPartitionedCall?merge_5/StatefulPartitionedCall?merge_7/StatefulPartitionedCall?merge_8/StatefulPartitionedCall?opt_11/StatefulPartitionedCall?opt_12/StatefulPartitionedCall?opt_2/StatefulPartitionedCall?opt_3/StatefulPartitionedCall?opt_5/StatefulPartitionedCall?opt_6/StatefulPartitionedCall?opt_8/StatefulPartitionedCall?opt_9/StatefulPartitionedCall?pred/StatefulPartitionedCall?
opt_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_1_layer_call_and_return_conditional_losses_68880?
gsc_1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_1_layer_call_and_return_conditional_losses_68890?
opt_2/StatefulPartitionedCallStatefulPartitionedCallopt_1/PartitionedCall:output:0opt_2_70246opt_2_70248*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_2_layer_call_and_return_conditional_losses_68903?
gsc_2/StatefulPartitionedCallStatefulPartitionedCallgsc_1/PartitionedCall:output:0gsc_2_70251gsc_2_70253*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_2_layer_call_and_return_conditional_losses_68920?
opt_3/StatefulPartitionedCallStatefulPartitionedCall&opt_2/StatefulPartitionedCall:output:0opt_3_70256opt_3_70258*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_3_layer_call_and_return_conditional_losses_68937?
gsc_3/StatefulPartitionedCallStatefulPartitionedCall&gsc_2/StatefulPartitionedCall:output:0gsc_3_70261gsc_3_70263*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_3_layer_call_and_return_conditional_losses_68954?
opt_4/PartitionedCallPartitionedCall&opt_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_4_layer_call_and_return_conditional_losses_68742?
gsc_4/PartitionedCallPartitionedCall&gsc_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_4_layer_call_and_return_conditional_losses_68730?
opt_5/StatefulPartitionedCallStatefulPartitionedCallopt_4/PartitionedCall:output:0opt_5_70268opt_5_70270*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_5_layer_call_and_return_conditional_losses_68973?
gsc_5/StatefulPartitionedCallStatefulPartitionedCallgsc_4/PartitionedCall:output:0gsc_5_70273gsc_5_70275*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_5_layer_call_and_return_conditional_losses_68990?
opt_6/StatefulPartitionedCallStatefulPartitionedCall&opt_5/StatefulPartitionedCall:output:0opt_6_70278opt_6_70280*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_6_layer_call_and_return_conditional_losses_69007?
gsc_6/StatefulPartitionedCallStatefulPartitionedCall&gsc_5/StatefulPartitionedCall:output:0gsc_6_70283gsc_6_70285*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_6_layer_call_and_return_conditional_losses_69024?
opt_7/PartitionedCallPartitionedCall&opt_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_7_layer_call_and_return_conditional_losses_68766?
gsc_7/PartitionedCallPartitionedCall&gsc_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_7_layer_call_and_return_conditional_losses_68754?
opt_8/StatefulPartitionedCallStatefulPartitionedCallopt_7/PartitionedCall:output:0opt_8_70290opt_8_70292*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_8_layer_call_and_return_conditional_losses_69043?
gsc_8/StatefulPartitionedCallStatefulPartitionedCallgsc_7/PartitionedCall:output:0gsc_8_70295gsc_8_70297*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_8_layer_call_and_return_conditional_losses_69060?
opt_9/StatefulPartitionedCallStatefulPartitionedCall&opt_8/StatefulPartitionedCall:output:0opt_9_70300opt_9_70302*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_opt_9_layer_call_and_return_conditional_losses_69077?
gsc_9/StatefulPartitionedCallStatefulPartitionedCall&gsc_8/StatefulPartitionedCall:output:0gsc_9_70305gsc_9_70307*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@88 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_gsc_9_layer_call_and_return_conditional_losses_69094?
opt_10/PartitionedCallPartitionedCall&opt_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_10_layer_call_and_return_conditional_losses_68790?
gsc_10/PartitionedCallPartitionedCall&gsc_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_10_layer_call_and_return_conditional_losses_68778?
opt_11/StatefulPartitionedCallStatefulPartitionedCallopt_10/PartitionedCall:output:0opt_11_70312opt_11_70314*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_11_layer_call_and_return_conditional_losses_69113?
gsc_11/StatefulPartitionedCallStatefulPartitionedCallgsc_10/PartitionedCall:output:0gsc_11_70317gsc_11_70319*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_11_layer_call_and_return_conditional_losses_69130?
opt_12/StatefulPartitionedCallStatefulPartitionedCall'opt_11/StatefulPartitionedCall:output:0opt_12_70322opt_12_70324*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_12_layer_call_and_return_conditional_losses_69147?
gsc_12/StatefulPartitionedCallStatefulPartitionedCall'gsc_11/StatefulPartitionedCall:output:0gsc_12_70327gsc_12_70329*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_12_layer_call_and_return_conditional_losses_69164?
gsc_13/PartitionedCallPartitionedCall'gsc_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_gsc_13_layer_call_and_return_conditional_losses_68802?
opt_13/PartitionedCallPartitionedCall'opt_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_opt_13_layer_call_and_return_conditional_losses_68814?
fuse_1/PartitionedCallPartitionedCallgsc_13/PartitionedCall:output:0opt_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_fuse_1_layer_call_and_return_conditional_losses_69178?
fuse_2/PartitionedCallPartitionedCallfuse_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_fuse_2_layer_call_and_return_conditional_losses_68826?
merge_1/StatefulPartitionedCallStatefulPartitionedCallfuse_2/PartitionedCall:output:0merge_1_70336merge_1_70338*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_1_layer_call_and_return_conditional_losses_69192?
merge_2/StatefulPartitionedCallStatefulPartitionedCall(merge_1/StatefulPartitionedCall:output:0merge_2_70341merge_2_70343*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_2_layer_call_and_return_conditional_losses_69209?
merge_3/PartitionedCallPartitionedCall(merge_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_3_layer_call_and_return_conditional_losses_68838?
merge_4/StatefulPartitionedCallStatefulPartitionedCall merge_3/PartitionedCall:output:0merge_4_70347merge_4_70349*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_4_layer_call_and_return_conditional_losses_69227?
merge_5/StatefulPartitionedCallStatefulPartitionedCall(merge_4/StatefulPartitionedCall:output:0merge_5_70352merge_5_70354*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_5_layer_call_and_return_conditional_losses_69244?
merge_6/PartitionedCallPartitionedCall(merge_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_6_layer_call_and_return_conditional_losses_68850?
merge_7/StatefulPartitionedCallStatefulPartitionedCall merge_6/PartitionedCall:output:0merge_7_70358merge_7_70360*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_7_layer_call_and_return_conditional_losses_69262?
merge_8/StatefulPartitionedCallStatefulPartitionedCall(merge_7/StatefulPartitionedCall:output:0merge_8_70363merge_8_70365*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_8_layer_call_and_return_conditional_losses_69279?
merge_9/PartitionedCallPartitionedCall(merge_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_merge_9_layer_call_and_return_conditional_losses_68862?
flat/PartitionedCallPartitionedCall merge_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_flat_layer_call_and_return_conditional_losses_69292?
fc_1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0
fc_1_70370
fc_1_70372*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_1_layer_call_and_return_conditional_losses_69305?
fc_2/PartitionedCallPartitionedCall%fc_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_2_layer_call_and_return_conditional_losses_69316?
fc_3/StatefulPartitionedCallStatefulPartitionedCallfc_2/PartitionedCall:output:0
fc_3_70376
fc_3_70378*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_fc_3_layer_call_and_return_conditional_losses_69329?
pred/StatefulPartitionedCallStatefulPartitionedCall%fc_3/StatefulPartitionedCall:output:0
pred_70381
pred_70383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_pred_layer_call_and_return_conditional_losses_69346t
IdentityIdentity%pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^fc_1/StatefulPartitionedCall^fc_3/StatefulPartitionedCall^gsc_11/StatefulPartitionedCall^gsc_12/StatefulPartitionedCall^gsc_2/StatefulPartitionedCall^gsc_3/StatefulPartitionedCall^gsc_5/StatefulPartitionedCall^gsc_6/StatefulPartitionedCall^gsc_8/StatefulPartitionedCall^gsc_9/StatefulPartitionedCall ^merge_1/StatefulPartitionedCall ^merge_2/StatefulPartitionedCall ^merge_4/StatefulPartitionedCall ^merge_5/StatefulPartitionedCall ^merge_7/StatefulPartitionedCall ^merge_8/StatefulPartitionedCall^opt_11/StatefulPartitionedCall^opt_12/StatefulPartitionedCall^opt_2/StatefulPartitionedCall^opt_3/StatefulPartitionedCall^opt_5/StatefulPartitionedCall^opt_6/StatefulPartitionedCall^opt_8/StatefulPartitionedCall^opt_9/StatefulPartitionedCall^pred/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????@??: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2@
gsc_11/StatefulPartitionedCallgsc_11/StatefulPartitionedCall2@
gsc_12/StatefulPartitionedCallgsc_12/StatefulPartitionedCall2>
gsc_2/StatefulPartitionedCallgsc_2/StatefulPartitionedCall2>
gsc_3/StatefulPartitionedCallgsc_3/StatefulPartitionedCall2>
gsc_5/StatefulPartitionedCallgsc_5/StatefulPartitionedCall2>
gsc_6/StatefulPartitionedCallgsc_6/StatefulPartitionedCall2>
gsc_8/StatefulPartitionedCallgsc_8/StatefulPartitionedCall2>
gsc_9/StatefulPartitionedCallgsc_9/StatefulPartitionedCall2B
merge_1/StatefulPartitionedCallmerge_1/StatefulPartitionedCall2B
merge_2/StatefulPartitionedCallmerge_2/StatefulPartitionedCall2B
merge_4/StatefulPartitionedCallmerge_4/StatefulPartitionedCall2B
merge_5/StatefulPartitionedCallmerge_5/StatefulPartitionedCall2B
merge_7/StatefulPartitionedCallmerge_7/StatefulPartitionedCall2B
merge_8/StatefulPartitionedCallmerge_8/StatefulPartitionedCall2@
opt_11/StatefulPartitionedCallopt_11/StatefulPartitionedCall2@
opt_12/StatefulPartitionedCallopt_12/StatefulPartitionedCall2>
opt_2/StatefulPartitionedCallopt_2/StatefulPartitionedCall2>
opt_3/StatefulPartitionedCallopt_3/StatefulPartitionedCall2>
opt_5/StatefulPartitionedCallopt_5/StatefulPartitionedCall2>
opt_6/StatefulPartitionedCallopt_6/StatefulPartitionedCall2>
opt_8/StatefulPartitionedCallopt_8/StatefulPartitionedCall2>
opt_9/StatefulPartitionedCallopt_9/StatefulPartitionedCall2<
pred/StatefulPartitionedCallpred/StatefulPartitionedCall:^ Z
5
_output_shapes#
!:?????????@??
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
input_1>
serving_default_input_1:0?????????@??8
pred0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer-20
layer_with_weights-12
layer-21
layer_with_weights-13
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer-25
layer-26
layer-27
layer-28
layer_with_weights-16
layer-29
layer_with_weights-17
layer-30
 layer-31
!layer_with_weights-18
!layer-32
"layer_with_weights-19
"layer-33
#layer-34
$layer_with_weights-20
$layer-35
%layer_with_weights-21
%layer-36
&layer-37
'layer-38
(layer_with_weights-22
(layer-39
)layer-40
*layer_with_weights-23
*layer-41
+layer_with_weights-24
+layer-42
,	optimizer
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3_default_save_signature
4
signatures"
_tf_keras_network
"
_tf_keras_input_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Qkernel
Rbias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
?
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
?
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
?

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ukernel
vbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
?

}kernel
~bias
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?	
	?iter
?beta_1
?beta_2

?decay
?learning_rateAm?Bm?Im?Jm?Qm?Rm?Ym?Zm?mm?nm?um?vm?}m?~m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Av?Bv?Iv?Jv?Qv?Rv?Yv?Zv?mv?nv?uv?vv?}v?~v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
?
A0
B1
I2
J3
Q4
R5
Y6
Z7
m8
n9
u10
v11
}12
~13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49"
trackable_list_wrapper
?
A0
B1
I2
J3
Q4
R5
Y6
Z7
m8
n9
u10
v11
}12
~13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
3_default_save_signature
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_cnn_base_layer_call_fn_69456
(__inference_cnn_base_layer_call_fn_70644
(__inference_cnn_base_layer_call_fn_70749
(__inference_cnn_base_layer_call_fn_70241?
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
?2?
C__inference_cnn_base_layer_call_and_return_conditional_losses_70952
C__inference_cnn_base_layer_call_and_return_conditional_losses_71162
C__inference_cnn_base_layer_call_and_return_conditional_losses_70387
C__inference_cnn_base_layer_call_and_return_conditional_losses_70533?
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
 __inference__wrapped_model_68721input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_gsc_1_layer_call_fn_71274
%__inference_gsc_1_layer_call_fn_71279?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_gsc_1_layer_call_and_return_conditional_losses_71287
@__inference_gsc_1_layer_call_and_return_conditional_losses_71295?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_opt_1_layer_call_fn_71300
%__inference_opt_1_layer_call_fn_71305?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_opt_1_layer_call_and_return_conditional_losses_71313
@__inference_opt_1_layer_call_and_return_conditional_losses_71321?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
*:(2gsc_2/kernel
:2
gsc_2/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_gsc_2_layer_call_fn_71330?
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
@__inference_gsc_2_layer_call_and_return_conditional_losses_71341?
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
*:(2opt_2/kernel
:2
opt_2/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_opt_2_layer_call_fn_71350?
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
@__inference_opt_2_layer_call_and_return_conditional_losses_71361?
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
*:(2gsc_3/kernel
:2
gsc_3/bias
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_gsc_3_layer_call_fn_71370?
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
@__inference_gsc_3_layer_call_and_return_conditional_losses_71381?
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
*:(2opt_3/kernel
:2
opt_3/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_opt_3_layer_call_fn_71390?
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
@__inference_opt_3_layer_call_and_return_conditional_losses_71401?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_gsc_4_layer_call_fn_71406?
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
@__inference_gsc_4_layer_call_and_return_conditional_losses_71411?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_opt_4_layer_call_fn_71416?
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
@__inference_opt_4_layer_call_and_return_conditional_losses_71421?
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
*:(2gsc_5/kernel
:2
gsc_5/bias
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_gsc_5_layer_call_fn_71430?
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
@__inference_gsc_5_layer_call_and_return_conditional_losses_71441?
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
*:(2opt_5/kernel
:2
opt_5/bias
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_opt_5_layer_call_fn_71450?
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
@__inference_opt_5_layer_call_and_return_conditional_losses_71461?
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
*:(2gsc_6/kernel
:2
gsc_6/bias
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_gsc_6_layer_call_fn_71470?
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
@__inference_gsc_6_layer_call_and_return_conditional_losses_71481?
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
*:(2opt_6/kernel
:2
opt_6/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_opt_6_layer_call_fn_71490?
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
@__inference_opt_6_layer_call_and_return_conditional_losses_71501?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_gsc_7_layer_call_fn_71506?
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
@__inference_gsc_7_layer_call_and_return_conditional_losses_71511?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_opt_7_layer_call_fn_71516?
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
@__inference_opt_7_layer_call_and_return_conditional_losses_71521?
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
*:( 2gsc_8/kernel
: 2
gsc_8/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_gsc_8_layer_call_fn_71530?
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
@__inference_gsc_8_layer_call_and_return_conditional_losses_71541?
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
*:( 2opt_8/kernel
: 2
opt_8/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_opt_8_layer_call_fn_71550?
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
@__inference_opt_8_layer_call_and_return_conditional_losses_71561?
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
*:(  2gsc_9/kernel
: 2
gsc_9/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_gsc_9_layer_call_fn_71570?
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
@__inference_gsc_9_layer_call_and_return_conditional_losses_71581?
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
*:(  2opt_9/kernel
: 2
opt_9/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_opt_9_layer_call_fn_71590?
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
@__inference_opt_9_layer_call_and_return_conditional_losses_71601?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_gsc_10_layer_call_fn_71606?
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
A__inference_gsc_10_layer_call_and_return_conditional_losses_71611?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_opt_10_layer_call_fn_71616?
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
A__inference_opt_10_layer_call_and_return_conditional_losses_71621?
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
+:)  2gsc_11/kernel
: 2gsc_11/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_gsc_11_layer_call_fn_71630?
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
A__inference_gsc_11_layer_call_and_return_conditional_losses_71641?
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
+:)  2opt_11/kernel
: 2opt_11/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_opt_11_layer_call_fn_71650?
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
A__inference_opt_11_layer_call_and_return_conditional_losses_71661?
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
+:)  2gsc_12/kernel
: 2gsc_12/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_gsc_12_layer_call_fn_71670?
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
A__inference_gsc_12_layer_call_and_return_conditional_losses_71681?
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
+:)  2opt_12/kernel
: 2opt_12/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_opt_12_layer_call_fn_71690?
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
A__inference_opt_12_layer_call_and_return_conditional_losses_71701?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_gsc_13_layer_call_fn_71706?
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
A__inference_gsc_13_layer_call_and_return_conditional_losses_71711?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_opt_13_layer_call_fn_71716?
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
A__inference_opt_13_layer_call_and_return_conditional_losses_71721?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_fuse_1_layer_call_fn_71727?
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
A__inference_fuse_1_layer_call_and_return_conditional_losses_71733?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_fuse_2_layer_call_fn_71738?
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
A__inference_fuse_2_layer_call_and_return_conditional_losses_71743?
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
,:* @2merge_1/kernel
:@2merge_1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_merge_1_layer_call_fn_71752?
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
B__inference_merge_1_layer_call_and_return_conditional_losses_71763?
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
,:*@@2merge_2/kernel
:@2merge_2/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_merge_2_layer_call_fn_71772?
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
B__inference_merge_2_layer_call_and_return_conditional_losses_71783?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_merge_3_layer_call_fn_71788?
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
B__inference_merge_3_layer_call_and_return_conditional_losses_71793?
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
,:*@@2merge_4/kernel
:@2merge_4/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_merge_4_layer_call_fn_71802?
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
B__inference_merge_4_layer_call_and_return_conditional_losses_71813?
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
,:*@@2merge_5/kernel
:@2merge_5/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_merge_5_layer_call_fn_71822?
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
B__inference_merge_5_layer_call_and_return_conditional_losses_71833?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_merge_6_layer_call_fn_71838?
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
B__inference_merge_6_layer_call_and_return_conditional_losses_71843?
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
-:+@?2merge_7/kernel
:?2merge_7/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_merge_7_layer_call_fn_71852?
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
B__inference_merge_7_layer_call_and_return_conditional_losses_71863?
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
.:,??2merge_8/kernel
:?2merge_8/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_merge_8_layer_call_fn_71872?
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
B__inference_merge_8_layer_call_and_return_conditional_losses_71883?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_merge_9_layer_call_fn_71888?
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
B__inference_merge_9_layer_call_and_return_conditional_losses_71893?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
$__inference_flat_layer_call_fn_71898?
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
?__inference_flat_layer_call_and_return_conditional_losses_71904?
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
:
??2fc_1/kernel
:?2	fc_1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
$__inference_fc_1_layer_call_fn_71913?
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
?__inference_fc_1_layer_call_and_return_conditional_losses_71924?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
$__inference_fc_2_layer_call_fn_71929
$__inference_fc_2_layer_call_fn_71934?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_fc_2_layer_call_and_return_conditional_losses_71939
?__inference_fc_2_layer_call_and_return_conditional_losses_71951?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	? 2fc_3/kernel
: 2	fc_3/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
$__inference_fc_3_layer_call_fn_71960?
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
?__inference_fc_3_layer_call_and_return_conditional_losses_71971?
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
: 2pred/kernel
:2	pred/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
$__inference_pred_layer_call_fn_71980?
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
?__inference_pred_layer_call_and_return_conditional_losses_71991?
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
#__inference_signature_wrapper_71269input_1"?
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
v
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api"
_tf_keras_metric
v
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-2Adam/gsc_2/kernel/m
:2Adam/gsc_2/bias/m
/:-2Adam/opt_2/kernel/m
:2Adam/opt_2/bias/m
/:-2Adam/gsc_3/kernel/m
:2Adam/gsc_3/bias/m
/:-2Adam/opt_3/kernel/m
:2Adam/opt_3/bias/m
/:-2Adam/gsc_5/kernel/m
:2Adam/gsc_5/bias/m
/:-2Adam/opt_5/kernel/m
:2Adam/opt_5/bias/m
/:-2Adam/gsc_6/kernel/m
:2Adam/gsc_6/bias/m
/:-2Adam/opt_6/kernel/m
:2Adam/opt_6/bias/m
/:- 2Adam/gsc_8/kernel/m
: 2Adam/gsc_8/bias/m
/:- 2Adam/opt_8/kernel/m
: 2Adam/opt_8/bias/m
/:-  2Adam/gsc_9/kernel/m
: 2Adam/gsc_9/bias/m
/:-  2Adam/opt_9/kernel/m
: 2Adam/opt_9/bias/m
0:.  2Adam/gsc_11/kernel/m
: 2Adam/gsc_11/bias/m
0:.  2Adam/opt_11/kernel/m
: 2Adam/opt_11/bias/m
0:.  2Adam/gsc_12/kernel/m
: 2Adam/gsc_12/bias/m
0:.  2Adam/opt_12/kernel/m
: 2Adam/opt_12/bias/m
1:/ @2Adam/merge_1/kernel/m
:@2Adam/merge_1/bias/m
1:/@@2Adam/merge_2/kernel/m
:@2Adam/merge_2/bias/m
1:/@@2Adam/merge_4/kernel/m
:@2Adam/merge_4/bias/m
1:/@@2Adam/merge_5/kernel/m
:@2Adam/merge_5/bias/m
2:0@?2Adam/merge_7/kernel/m
 :?2Adam/merge_7/bias/m
3:1??2Adam/merge_8/kernel/m
 :?2Adam/merge_8/bias/m
$:"
??2Adam/fc_1/kernel/m
:?2Adam/fc_1/bias/m
#:!	? 2Adam/fc_3/kernel/m
: 2Adam/fc_3/bias/m
":  2Adam/pred/kernel/m
:2Adam/pred/bias/m
/:-2Adam/gsc_2/kernel/v
:2Adam/gsc_2/bias/v
/:-2Adam/opt_2/kernel/v
:2Adam/opt_2/bias/v
/:-2Adam/gsc_3/kernel/v
:2Adam/gsc_3/bias/v
/:-2Adam/opt_3/kernel/v
:2Adam/opt_3/bias/v
/:-2Adam/gsc_5/kernel/v
:2Adam/gsc_5/bias/v
/:-2Adam/opt_5/kernel/v
:2Adam/opt_5/bias/v
/:-2Adam/gsc_6/kernel/v
:2Adam/gsc_6/bias/v
/:-2Adam/opt_6/kernel/v
:2Adam/opt_6/bias/v
/:- 2Adam/gsc_8/kernel/v
: 2Adam/gsc_8/bias/v
/:- 2Adam/opt_8/kernel/v
: 2Adam/opt_8/bias/v
/:-  2Adam/gsc_9/kernel/v
: 2Adam/gsc_9/bias/v
/:-  2Adam/opt_9/kernel/v
: 2Adam/opt_9/bias/v
0:.  2Adam/gsc_11/kernel/v
: 2Adam/gsc_11/bias/v
0:.  2Adam/opt_11/kernel/v
: 2Adam/opt_11/bias/v
0:.  2Adam/gsc_12/kernel/v
: 2Adam/gsc_12/bias/v
0:.  2Adam/opt_12/kernel/v
: 2Adam/opt_12/bias/v
1:/ @2Adam/merge_1/kernel/v
:@2Adam/merge_1/bias/v
1:/@@2Adam/merge_2/kernel/v
:@2Adam/merge_2/bias/v
1:/@@2Adam/merge_4/kernel/v
:@2Adam/merge_4/bias/v
1:/@@2Adam/merge_5/kernel/v
:@2Adam/merge_5/bias/v
2:0@?2Adam/merge_7/kernel/v
 :?2Adam/merge_7/bias/v
3:1??2Adam/merge_8/kernel/v
 :?2Adam/merge_8/bias/v
$:"
??2Adam/fc_1/kernel/v
:?2Adam/fc_1/bias/v
#:!	? 2Adam/fc_3/kernel/v
: 2Adam/fc_3/bias/v
":  2Adam/pred/kernel/v
:2Adam/pred/bias/v?
 __inference__wrapped_model_68721?VIJABYZQRuvmn??}~??????????????????????????????????>?;
4?1
/?,
input_1?????????@??
? "+?(
&
pred?
pred??????????
C__inference_cnn_base_layer_call_and_return_conditional_losses_70387?VIJABYZQRuvmn??}~??????????????????????????????????F?C
<?9
/?,
input_1?????????@??
p 

 
? "%?"
?
0?????????
? ?
C__inference_cnn_base_layer_call_and_return_conditional_losses_70533?VIJABYZQRuvmn??}~??????????????????????????????????F?C
<?9
/?,
input_1?????????@??
p

 
? "%?"
?
0?????????
? ?
C__inference_cnn_base_layer_call_and_return_conditional_losses_70952?VIJABYZQRuvmn??}~??????????????????????????????????E?B
;?8
.?+
inputs?????????@??
p 

 
? "%?"
?
0?????????
? ?
C__inference_cnn_base_layer_call_and_return_conditional_losses_71162?VIJABYZQRuvmn??}~??????????????????????????????????E?B
;?8
.?+
inputs?????????@??
p

 
? "%?"
?
0?????????
? ?
(__inference_cnn_base_layer_call_fn_69456?VIJABYZQRuvmn??}~??????????????????????????????????F?C
<?9
/?,
input_1?????????@??
p 

 
? "???????????
(__inference_cnn_base_layer_call_fn_70241?VIJABYZQRuvmn??}~??????????????????????????????????F?C
<?9
/?,
input_1?????????@??
p

 
? "???????????
(__inference_cnn_base_layer_call_fn_70644?VIJABYZQRuvmn??}~??????????????????????????????????E?B
;?8
.?+
inputs?????????@??
p 

 
? "???????????
(__inference_cnn_base_layer_call_fn_70749?VIJABYZQRuvmn??}~??????????????????????????????????E?B
;?8
.?+
inputs?????????@??
p

 
? "???????????
?__inference_fc_1_layer_call_and_return_conditional_losses_71924`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
$__inference_fc_1_layer_call_fn_71913S??0?-
&?#
!?
inputs??????????
? "????????????
?__inference_fc_2_layer_call_and_return_conditional_losses_71939^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
?__inference_fc_2_layer_call_and_return_conditional_losses_71951^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? y
$__inference_fc_2_layer_call_fn_71929Q4?1
*?'
!?
inputs??????????
p 
? "???????????y
$__inference_fc_2_layer_call_fn_71934Q4?1
*?'
!?
inputs??????????
p
? "????????????
?__inference_fc_3_layer_call_and_return_conditional_losses_71971_??0?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? z
$__inference_fc_3_layer_call_fn_71960R??0?-
&?#
!?
inputs??????????
? "?????????? ?
?__inference_flat_layer_call_and_return_conditional_losses_71904f<?9
2?/
-?*
inputs??????????
? "&?#
?
0??????????
? ?
$__inference_flat_layer_call_fn_71898Y<?9
2?/
-?*
inputs??????????
? "????????????
A__inference_fuse_1_layer_call_and_return_conditional_losses_71733?r?o
h?e
c?`
.?+
inputs/0?????????@ 
.?+
inputs/1?????????@ 
? "1?.
'?$
0?????????@ 
? ?
&__inference_fuse_1_layer_call_fn_71727?r?o
h?e
c?`
.?+
inputs/0?????????@ 
.?+
inputs/1?????????@ 
? "$?!?????????@ ?
A__inference_fuse_2_layer_call_and_return_conditional_losses_71743?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
&__inference_fuse_2_layer_call_fn_71738?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
A__inference_gsc_10_layer_call_and_return_conditional_losses_71611?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
&__inference_gsc_10_layer_call_fn_71606?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
A__inference_gsc_11_layer_call_and_return_conditional_losses_71641v??;?8
1?.
,?)
inputs?????????@ 
? "1?.
'?$
0?????????@ 
? ?
&__inference_gsc_11_layer_call_fn_71630i??;?8
1?.
,?)
inputs?????????@ 
? "$?!?????????@ ?
A__inference_gsc_12_layer_call_and_return_conditional_losses_71681v??;?8
1?.
,?)
inputs?????????@ 
? "1?.
'?$
0?????????@ 
? ?
&__inference_gsc_12_layer_call_fn_71670i??;?8
1?.
,?)
inputs?????????@ 
? "$?!?????????@ ?
A__inference_gsc_13_layer_call_and_return_conditional_losses_71711?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
&__inference_gsc_13_layer_call_fn_71706?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
@__inference_gsc_1_layer_call_and_return_conditional_losses_71287|E?B
;?8
.?+
inputs?????????@??

 
p 
? "3?0
)?&
0?????????@??
? ?
@__inference_gsc_1_layer_call_and_return_conditional_losses_71295|E?B
;?8
.?+
inputs?????????@??

 
p
? "3?0
)?&
0?????????@??
? ?
%__inference_gsc_1_layer_call_fn_71274oE?B
;?8
.?+
inputs?????????@??

 
p 
? "&?#?????????@???
%__inference_gsc_1_layer_call_fn_71279oE?B
;?8
.?+
inputs?????????@??

 
p
? "&?#?????????@???
@__inference_gsc_2_layer_call_and_return_conditional_losses_71341xAB=?:
3?0
.?+
inputs?????????@??
? "3?0
)?&
0?????????@??
? ?
%__inference_gsc_2_layer_call_fn_71330kAB=?:
3?0
.?+
inputs?????????@??
? "&?#?????????@???
@__inference_gsc_3_layer_call_and_return_conditional_losses_71381xQR=?:
3?0
.?+
inputs?????????@??
? "3?0
)?&
0?????????@??
? ?
%__inference_gsc_3_layer_call_fn_71370kQR=?:
3?0
.?+
inputs?????????@??
? "&?#?????????@???
@__inference_gsc_4_layer_call_and_return_conditional_losses_71411?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
%__inference_gsc_4_layer_call_fn_71406?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
@__inference_gsc_5_layer_call_and_return_conditional_losses_71441tmn;?8
1?.
,?)
inputs?????????@pp
? "1?.
'?$
0?????????@pp
? ?
%__inference_gsc_5_layer_call_fn_71430gmn;?8
1?.
,?)
inputs?????????@pp
? "$?!?????????@pp?
@__inference_gsc_6_layer_call_and_return_conditional_losses_71481t}~;?8
1?.
,?)
inputs?????????@pp
? "1?.
'?$
0?????????@pp
? ?
%__inference_gsc_6_layer_call_fn_71470g}~;?8
1?.
,?)
inputs?????????@pp
? "$?!?????????@pp?
@__inference_gsc_7_layer_call_and_return_conditional_losses_71511?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
%__inference_gsc_7_layer_call_fn_71506?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
@__inference_gsc_8_layer_call_and_return_conditional_losses_71541v??;?8
1?.
,?)
inputs?????????@88
? "1?.
'?$
0?????????@88 
? ?
%__inference_gsc_8_layer_call_fn_71530i??;?8
1?.
,?)
inputs?????????@88
? "$?!?????????@88 ?
@__inference_gsc_9_layer_call_and_return_conditional_losses_71581v??;?8
1?.
,?)
inputs?????????@88 
? "1?.
'?$
0?????????@88 
? ?
%__inference_gsc_9_layer_call_fn_71570i??;?8
1?.
,?)
inputs?????????@88 
? "$?!?????????@88 ?
B__inference_merge_1_layer_call_and_return_conditional_losses_71763v??;?8
1?.
,?)
inputs????????? 
? "1?.
'?$
0?????????@
? ?
'__inference_merge_1_layer_call_fn_71752i??;?8
1?.
,?)
inputs????????? 
? "$?!?????????@?
B__inference_merge_2_layer_call_and_return_conditional_losses_71783v??;?8
1?.
,?)
inputs?????????@
? "1?.
'?$
0?????????@
? ?
'__inference_merge_2_layer_call_fn_71772i??;?8
1?.
,?)
inputs?????????@
? "$?!?????????@?
B__inference_merge_3_layer_call_and_return_conditional_losses_71793?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
'__inference_merge_3_layer_call_fn_71788?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
B__inference_merge_4_layer_call_and_return_conditional_losses_71813v??;?8
1?.
,?)
inputs?????????@
? "1?.
'?$
0?????????@
? ?
'__inference_merge_4_layer_call_fn_71802i??;?8
1?.
,?)
inputs?????????@
? "$?!?????????@?
B__inference_merge_5_layer_call_and_return_conditional_losses_71833v??;?8
1?.
,?)
inputs?????????@
? "1?.
'?$
0?????????@
? ?
'__inference_merge_5_layer_call_fn_71822i??;?8
1?.
,?)
inputs?????????@
? "$?!?????????@?
B__inference_merge_6_layer_call_and_return_conditional_losses_71843?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
'__inference_merge_6_layer_call_fn_71838?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
B__inference_merge_7_layer_call_and_return_conditional_losses_71863w??;?8
1?.
,?)
inputs?????????@
? "2?/
(?%
0??????????
? ?
'__inference_merge_7_layer_call_fn_71852j??;?8
1?.
,?)
inputs?????????@
? "%?"???????????
B__inference_merge_8_layer_call_and_return_conditional_losses_71883x??<?9
2?/
-?*
inputs??????????
? "2?/
(?%
0??????????
? ?
'__inference_merge_8_layer_call_fn_71872k??<?9
2?/
-?*
inputs??????????
? "%?"???????????
B__inference_merge_9_layer_call_and_return_conditional_losses_71893?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
'__inference_merge_9_layer_call_fn_71888?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
A__inference_opt_10_layer_call_and_return_conditional_losses_71621?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
&__inference_opt_10_layer_call_fn_71616?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
A__inference_opt_11_layer_call_and_return_conditional_losses_71661v??;?8
1?.
,?)
inputs?????????@ 
? "1?.
'?$
0?????????@ 
? ?
&__inference_opt_11_layer_call_fn_71650i??;?8
1?.
,?)
inputs?????????@ 
? "$?!?????????@ ?
A__inference_opt_12_layer_call_and_return_conditional_losses_71701v??;?8
1?.
,?)
inputs?????????@ 
? "1?.
'?$
0?????????@ 
? ?
&__inference_opt_12_layer_call_fn_71690i??;?8
1?.
,?)
inputs?????????@ 
? "$?!?????????@ ?
A__inference_opt_13_layer_call_and_return_conditional_losses_71721?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
&__inference_opt_13_layer_call_fn_71716?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
@__inference_opt_1_layer_call_and_return_conditional_losses_71313|E?B
;?8
.?+
inputs?????????@??

 
p 
? "3?0
)?&
0?????????@??
? ?
@__inference_opt_1_layer_call_and_return_conditional_losses_71321|E?B
;?8
.?+
inputs?????????@??

 
p
? "3?0
)?&
0?????????@??
? ?
%__inference_opt_1_layer_call_fn_71300oE?B
;?8
.?+
inputs?????????@??

 
p 
? "&?#?????????@???
%__inference_opt_1_layer_call_fn_71305oE?B
;?8
.?+
inputs?????????@??

 
p
? "&?#?????????@???
@__inference_opt_2_layer_call_and_return_conditional_losses_71361xIJ=?:
3?0
.?+
inputs?????????@??
? "3?0
)?&
0?????????@??
? ?
%__inference_opt_2_layer_call_fn_71350kIJ=?:
3?0
.?+
inputs?????????@??
? "&?#?????????@???
@__inference_opt_3_layer_call_and_return_conditional_losses_71401xYZ=?:
3?0
.?+
inputs?????????@??
? "3?0
)?&
0?????????@??
? ?
%__inference_opt_3_layer_call_fn_71390kYZ=?:
3?0
.?+
inputs?????????@??
? "&?#?????????@???
@__inference_opt_4_layer_call_and_return_conditional_losses_71421?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
%__inference_opt_4_layer_call_fn_71416?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
@__inference_opt_5_layer_call_and_return_conditional_losses_71461tuv;?8
1?.
,?)
inputs?????????@pp
? "1?.
'?$
0?????????@pp
? ?
%__inference_opt_5_layer_call_fn_71450guv;?8
1?.
,?)
inputs?????????@pp
? "$?!?????????@pp?
@__inference_opt_6_layer_call_and_return_conditional_losses_71501v??;?8
1?.
,?)
inputs?????????@pp
? "1?.
'?$
0?????????@pp
? ?
%__inference_opt_6_layer_call_fn_71490i??;?8
1?.
,?)
inputs?????????@pp
? "$?!?????????@pp?
@__inference_opt_7_layer_call_and_return_conditional_losses_71521?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
%__inference_opt_7_layer_call_fn_71516?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
@__inference_opt_8_layer_call_and_return_conditional_losses_71561v??;?8
1?.
,?)
inputs?????????@88
? "1?.
'?$
0?????????@88 
? ?
%__inference_opt_8_layer_call_fn_71550i??;?8
1?.
,?)
inputs?????????@88
? "$?!?????????@88 ?
@__inference_opt_9_layer_call_and_return_conditional_losses_71601v??;?8
1?.
,?)
inputs?????????@88 
? "1?.
'?$
0?????????@88 
? ?
%__inference_opt_9_layer_call_fn_71590i??;?8
1?.
,?)
inputs?????????@88 
? "$?!?????????@88 ?
?__inference_pred_layer_call_and_return_conditional_losses_71991^??/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? y
$__inference_pred_layer_call_fn_71980Q??/?,
%?"
 ?
inputs????????? 
? "???????????
#__inference_signature_wrapper_71269?VIJABYZQRuvmn??}~??????????????????????????????????I?F
? 
??<
:
input_1/?,
input_1?????????@??"+?(
&
pred?
pred?????????