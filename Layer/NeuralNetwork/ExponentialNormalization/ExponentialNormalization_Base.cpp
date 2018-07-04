//======================================
// �o�b�`���K�����C���[
//======================================
#include"stdafx.h"

#include"ExponentialNormalization_FUNC.hpp"

#include"ExponentialNormalization_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
ExponentialNormalization_Base::ExponentialNormalization_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateRuntimeParameter(), i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
ExponentialNormalization_Base::~ExponentialNormalization_Base()
{
}
