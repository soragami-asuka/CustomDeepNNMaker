//======================================
// �o�͐M���������C���[
//======================================
#include"stdafx.h"

#include"SignalArray2Value_FUNC.hpp"

#include"SignalArray2Value_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
SignalArray2Value_Base::SignalArray2Value_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
SignalArray2Value_Base::~SignalArray2Value_Base()
{
}
