//======================================
// �o�͐M���������C���[
//======================================
#include"stdafx.h"

#include"Value2SignalArray_FUNC.hpp"

#include"Value2SignalArray_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
Value2SignalArray_Base::Value2SignalArray_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
Value2SignalArray_Base::~Value2SignalArray_Base()
{
}
