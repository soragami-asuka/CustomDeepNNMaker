//======================================
// �����֐����C���[
//======================================
#include"stdafx.h"

#include"MergeAdd_FUNC.hpp"

#include"MergeAdd_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
MergeAdd_Base::MergeAdd_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNMult2SingleLayerBase(guid, i_lpInputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
MergeAdd_Base::~MergeAdd_Base()
{
}

