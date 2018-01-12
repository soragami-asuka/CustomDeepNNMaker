//======================================
// �����֐����C���[
//======================================
#include"stdafx.h"

#include"MergeAverage_FUNC.hpp"

#include"MergeAverage_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
MergeAverage_Base::MergeAverage_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNMult2SingleLayerBase(guid, i_lpInputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
MergeAverage_Base::~MergeAverage_Base()
{
}

