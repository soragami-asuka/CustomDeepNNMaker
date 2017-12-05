//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#include"stdafx.h"

#include"FullyConnect_FUNC.hpp"

#include"FullyConnect_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
FullyConnect_Base::FullyConnect_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateRuntimeParameter(), i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
FullyConnect_Base::~FullyConnect_Base()
{
}


//===========================
// �ŗL�֐�
//===========================
/** �j���[���������擾���� */
U32 FullyConnect_Base::GetNeuronCount()const
{
	const FullyConnect_LayerData_Base* pLayerData = dynamic_cast<const FullyConnect_LayerData_Base*>(&this->GetLayerData());
	if(pLayerData == NULL)
		return NULL;

	return pLayerData->GetNeuronCount();
}
