//======================================
// �o�͐M���������C���[
//======================================
#include"stdafx.h"

#include"LimitBackPropagationRange_FUNC.hpp"

#include"LimitBackPropagationRange_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
LimitBackPropagationRange_Base::LimitBackPropagationRange_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
LimitBackPropagationRange_Base::~LimitBackPropagationRange_Base()
{
}
