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
FullyConnect_Base::FullyConnect_Base(Gravisbell::GUID guid)
	:	INNLayer()
	,	guid				(guid)
	,	pLearnData			(NULL)
{
}

/** �f�X�g���N�^ */
FullyConnect_Base::~FullyConnect_Base()
{
	if(pLearnData != NULL)
		delete pLearnData;
}


//===========================
// ���C���[����
//===========================

/** ���C���[��ʂ̎擾.
	ELayerKind �̑g�ݍ��킹. */
U32 FullyConnect_Base::GetLayerKindBase(void)const
{
	return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
}

/** ���C���[�ŗL��GUID���擾���� */
Gravisbell::GUID FullyConnect_Base::GetGUID(void)const
{
	return this->guid;
}

/** ���C���[���ʃR�[�h���擾����.
	@param o_layerCode	�i�[��o�b�t�@
	@return ���������ꍇ0 */
Gravisbell::GUID FullyConnect_Base::GetLayerCode(void)const
{
	return this->GetLayerData().GetLayerCode();
}

/** �o�b�`�T�C�Y���擾����.
	@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
unsigned int FullyConnect_Base::GetBatchSize()const
{
	return this->batchSize;
}

/** ���C���[�̐ݒ�����擾���� */
const SettingData::Standard::IData* FullyConnect_Base::GetLayerStructure()const
{
	return this->GetLayerData().GetLayerStructure();
}


//===========================
// ���̓��C���[�֘A
//===========================
/** ���̓f�[�^�\�����擾����.
	@return	���̓f�[�^�\�� */
IODataStruct FullyConnect_Base::GetInputDataStruct()const
{
	return this->GetLayerData().GetInputDataStruct();
}

/** ���̓o�b�t�@�����擾����. */
unsigned int FullyConnect_Base::GetInputBufferCount()const
{
	return this->GetLayerData().GetInputBufferCount();
}


//===========================
// �o�̓��C���[�֘A
//===========================
/** �o�̓f�[�^�\�����擾���� */
IODataStruct FullyConnect_Base::GetOutputDataStruct()const
{
	return this->GetLayerData().GetOutputDataStruct();
}

/** �o�̓o�b�t�@�����擾���� */
unsigned int FullyConnect_Base::GetOutputBufferCount()const
{
	return this->GetLayerData().GetOutputBufferCount();
}


//===========================
// �ŗL�֐�
//===========================
/** �j���[���������擾���� */
U32 FullyConnect_Base::GetNeuronCount()const
{
	return this->GetLayerData().GetNeuronCount();
}
