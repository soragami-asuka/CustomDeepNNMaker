//======================================
// �o�͐M���������C���[
//======================================
#include"stdafx.h"

#include"Reshape_SquaresCenterCross_FUNC.hpp"

#include"Reshape_SquaresCenterCross_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
Reshape_SquaresCenterCross_Base::Reshape_SquaresCenterCross_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
Reshape_SquaresCenterCross_Base::~Reshape_SquaresCenterCross_Base()
{
}
