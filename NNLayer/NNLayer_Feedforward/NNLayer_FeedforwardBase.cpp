//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#include"stdafx.h"

#include"NNLayer_Feedforward_FUNC.hpp"

#include"NNLayer_FeedforwardBase.h"

using namespace Gravisbell;
using namespace Gravisbell::NeuralNetwork;


/** �R���X�g���N�^ */
NNLayer_FeedforwardBase::NNLayer_FeedforwardBase(Gravisbell::GUID guid)
	:	INNLayer()
	,	guid			(guid)
	,	pLayerStructure	(NULL)
	,	pLearnData		(NULL)
	,	lppInputFromLayer	()		/**< ���͌����C���[�̃��X�g */
	,	lppOutputToLayer	()		/**< �o�͐惌�C���[�̃��X�g */
{
}

/** �f�X�g���N�^ */
NNLayer_FeedforwardBase::~NNLayer_FeedforwardBase()
{
	if(pLayerStructure != NULL)
		delete pLayerStructure;
	if(pLearnData != NULL)
		delete pLearnData;
}


//===========================
// ���C���[����
//===========================

/** ���C���[��ʂ̎擾.
	ELayerKind �̑g�ݍ��킹. */
unsigned int NNLayer_FeedforwardBase::GetLayerKindBase()const
{
	return Gravisbell::NeuralNetwork::LAYER_KIND_CALC | Gravisbell::NeuralNetwork::LAYER_KIND_SINGLE_INPUT | Gravisbell::NeuralNetwork::LAYER_KIND_SINGLE_OUTPUT;
}

/** ���C���[�ŗL��GUID���擾���� */
Gravisbell::ErrorCode NNLayer_FeedforwardBase::GetGUID(Gravisbell::GUID& o_guid)const
{
	o_guid = this->guid;

	return ERROR_CODE_NONE;
}

/** ���C���[���ʃR�[�h���擾����.
	@param o_layerCode	�i�[��o�b�t�@
	@return ���������ꍇ0 */
Gravisbell::ErrorCode NNLayer_FeedforwardBase::GetLayerCode(Gravisbell::GUID& o_layerCode)const
{
	return ::GetLayerCode(o_layerCode);
}

/** �o�b�`�T�C�Y���擾����.
	@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
unsigned int NNLayer_FeedforwardBase::GetBatchSize()const
{
	return this->batchSize;
}


/** �ݒ����ݒ� */
Gravisbell::ErrorCode NNLayer_FeedforwardBase::SetLayerConfig(const SettingData::Standard::IData& config)
{
	Gravisbell::ErrorCode err = ERROR_CODE_NONE;

	// ���C���[�R�[�h���m�F
	{
		GUID config_guid;
		err = config.GetLayerCode(config_guid);
		if(err != ERROR_CODE_NONE)
			return err;

		GUID layer_guid;
		err = ::GetLayerCode(layer_guid);
		if(err != ERROR_CODE_NONE)
			return err;

		if(config_guid != layer_guid)
			return ERROR_CODE_INITLAYER_DISAGREE_CONFIG;
	}

	if(this->pLayerStructure != NULL)
		delete this->pLayerStructure;
	this->pLayerStructure = config.Clone();

	// �\���̂ɓǂݍ���
	this->pLayerStructure->WriteToStruct((BYTE*)&this->layerStructure);

	return ERROR_CODE_NONE;
}
/** ���C���[�̐ݒ�����擾���� */
const SettingData::Standard::IData* NNLayer_FeedforwardBase::GetLayerConfig()const
{
	return this->pLayerStructure;
}


//===========================
// ���̓��C���[�֘A
//===========================
/** ���̓f�[�^�\�����擾����.
	@return	���̓f�[�^�\�� */
IODataStruct NNLayer_FeedforwardBase::GetInputDataStruct()const
{
	return this->inputDataStruct;
}

/** ���̓o�b�t�@�����擾����. */
unsigned int NNLayer_FeedforwardBase::GetInputBufferCount()const
{
	return this->inputDataStruct.x * this->inputDataStruct.y * this->inputDataStruct.z * this->inputDataStruct.ch;
}


//===========================
// ���C���[�ۑ�
//===========================
/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
unsigned int NNLayer_FeedforwardBase::GetUseBufferByteCount()const
{
	unsigned int bufferSize = 0;

	if(pLayerStructure == NULL)
		return 0;

	// �ݒ���
	bufferSize += pLayerStructure->GetUseBufferByteCount();

	// �{�̂̃o�C�g��
	bufferSize += (this->GetNeuronCount() * this->GetInputBufferCount()) * sizeof(NEURON_TYPE);	// �j���[�����W��
	bufferSize += this->GetNeuronCount() * sizeof(NEURON_TYPE);	// �o�C�A�X�W��


	return bufferSize;
}


//===========================
// �o�̓��C���[�֘A
//===========================
/** �o�̓f�[�^�\�����擾���� */
IODataStruct NNLayer_FeedforwardBase::GetOutputDataStruct()const
{
	IODataStruct outputDataStruct;

	outputDataStruct.x = 1;
	outputDataStruct.y = 1;
	outputDataStruct.z = 1;
	outputDataStruct.ch = this->GetNeuronCount();

	return outputDataStruct;
}

/** �o�̓o�b�t�@�����擾���� */
unsigned int NNLayer_FeedforwardBase::GetOutputBufferCount()const
{
	IODataStruct outputDataStruct = GetOutputDataStruct();

	return outputDataStruct.x * outputDataStruct.y * outputDataStruct.z * outputDataStruct.ch;
}


//===========================
// �ŗL�֐�
//===========================
/** �j���[���������擾���� */
unsigned int NNLayer_FeedforwardBase::GetNeuronCount()const
{
	return this->layerStructure.NeuronCount;
}