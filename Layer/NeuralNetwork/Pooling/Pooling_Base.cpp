//======================================
// �����֐����C���[
//======================================
#include"stdafx.h"

#include"Pooling_FUNC.hpp"

#include"Pooling_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
Pooling_Base::Pooling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
Pooling_Base::~Pooling_Base()
{
}

