//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̌������C���[
//======================================
#include"stdafx.h"

#include"UpSampling_FUNC.hpp"

#include"UpSampling_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
UpSampling_Base::UpSampling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
UpSampling_Base::~UpSampling_Base()
{
}

