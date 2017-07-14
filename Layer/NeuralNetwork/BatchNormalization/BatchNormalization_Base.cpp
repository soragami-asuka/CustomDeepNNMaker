//======================================
// �o�b�`���K�����C���[
//======================================
#include"stdafx.h"

#include"BatchNormalization_FUNC.hpp"

#include"BatchNormalization_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
BatchNormalization_Base::BatchNormalization_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
BatchNormalization_Base::~BatchNormalization_Base()
{
}

