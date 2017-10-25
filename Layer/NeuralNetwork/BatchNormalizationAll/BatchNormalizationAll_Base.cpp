//======================================
// �o�b�`���K�����C���[
//======================================
#include"stdafx.h"

#include"BatchNormalizationAll_FUNC.hpp"

#include"BatchNormalizationAll_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
BatchNormalizationAll_Base::BatchNormalizationAll_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateRuntimeParameter(), i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
BatchNormalizationAll_Base::~BatchNormalizationAll_Base()
{
}

