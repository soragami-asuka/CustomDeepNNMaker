//======================================
// �o�͐M���������C���[
//======================================
#include"stdafx.h"

#include"ChooseChannel_FUNC.hpp"

#include"ChooseChannel_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
ChooseChannel_Base::ChooseChannel_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
ChooseChannel_Base::~ChooseChannel_Base()
{
}
