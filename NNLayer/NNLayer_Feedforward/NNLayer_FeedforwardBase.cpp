//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#include"stdafx.h"

#include"NNLayer_Feedforward.h"
#include"NNLayer_FeedforwardBase.h"



/** �R���X�g���N�^ */
NNLayer_FeedforwardBase::NNLayer_FeedforwardBase(GUID guid)
	:	INNLayer()
	,	guid			(guid)
	,	pConfig			(NULL)
	,	lppInputFromLayer	()		/**< ���͌����C���[�̃��X�g */
	,	lppOutputToLayer()	/**< �o�͐惌�C���[�̃��X�g */
{
}

/** �f�X�g���N�^ */
NNLayer_FeedforwardBase::~NNLayer_FeedforwardBase()
{
	if(pConfig != NULL)
		delete pConfig;
}


//===========================
// ���C���[����
//===========================
/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
unsigned int NNLayer_FeedforwardBase::GetUseBufferByteCount()const
{
	unsigned int bufferSize = 0;

	if(pConfig == NULL)
		return 0;

	// �ݒ���
	bufferSize += pConfig->GetUseBufferByteCount();

	// �{�̂̃o�C�g��
	bufferSize += (this->GetNeuronCount() * this->GetInputBufferCount()) * sizeof(NEURON_TYPE);	// �j���[�����W��
	bufferSize += this->GetNeuronCount() * sizeof(NEURON_TYPE);	// �o�C�A�X�W��


	return bufferSize;
}

/** ���C���[�ŗL��GUID���擾���� */
ELayerErrorCode NNLayer_FeedforwardBase::GetGUID(GUID& o_guid)const
{
	o_guid = this->guid;

	return LAYER_ERROR_NONE;
}

/** ���C���[���ʃR�[�h���擾����.
	@param o_layerCode	�i�[��o�b�t�@
	@return ���������ꍇ0 */
ELayerErrorCode NNLayer_FeedforwardBase::GetLayerCode(GUID& o_layerCode)const
{
	return ::GetLayerCode(o_layerCode);
}

/** �ݒ����ݒ� */
ELayerErrorCode NNLayer_FeedforwardBase::SetLayerConfig(const INNLayerConfig& config)
{
	ELayerErrorCode err = LAYER_ERROR_NONE;

	// ���C���[�R�[�h���m�F
	{
		GUID config_guid;
		err = config.GetLayerCode(config_guid);
		if(err != LAYER_ERROR_NONE)
			return err;

		GUID layer_guid;
		err = ::GetLayerCode(layer_guid);
		if(err != LAYER_ERROR_NONE)
			return err;

		if(config_guid != layer_guid)
			return LAYER_ERROR_INITLAYER_DISAGREE_CONFIG;
	}

	if(this->pConfig != NULL)
		delete this->pConfig;
	this->pConfig = config.Clone();

	return LAYER_ERROR_NONE;
}
/** ���C���[�̐ݒ�����擾���� */
const INNLayerConfig* NNLayer_FeedforwardBase::GetLayerConfig()const
{
	return this->pConfig;
}


//===========================
// ���̓��C���[�֘A
//===========================
/** ���̓o�b�t�@�����擾����. */
unsigned int NNLayer_FeedforwardBase::GetInputBufferCount()const
{
	unsigned int intputBufferCount = 0;
	for(auto layer : this->lppInputFromLayer)
	{
		intputBufferCount += layer->GetOutputBufferCount();
	}
	return intputBufferCount;
}


/** ���͌����C���[�ւ̃����N��ǉ�����.
	@param	pLayer	�ǉ�������͌����C���[ */
ELayerErrorCode NNLayer_FeedforwardBase::AddInputFromLayer(IOutputLayer* pLayer)
{
	// �������̓��C���[�����݂��Ȃ��m�F����
	for(auto it : this->lppInputFromLayer)
	{
		if(it == pLayer)
			return LAYER_ERROR_ADDLAYER_ALREADY_SAMEID;
	}


	// ���X�g�ɒǉ�
	this->lppInputFromLayer.push_back(pLayer);

	// ���͌����C���[�Ɏ������o�͐�Ƃ��Ēǉ�
	pLayer->AddOutputToLayer(this);

	return LAYER_ERROR_NONE;
}
/** ���͌����C���[�ւ̃����N���폜����.
	@param	pLayer	�폜����o�͐惌�C���[ */
ELayerErrorCode NNLayer_FeedforwardBase::EraseInputFromLayer(IOutputLayer* pLayer)
{
	// ���X�g���猟�����č폜
	auto it = this->lppInputFromLayer.begin();
	while(it != this->lppInputFromLayer.end())
	{
		if(*it == pLayer)
		{
			// ���X�g����폜
			this->lppInputFromLayer.erase(it);

			// �폜���C���[�ɓo�^����Ă��鎩�����g���폜
			pLayer->EraseOutputToLayer(this);

			return LAYER_ERROR_NONE;
		}
		it++;
	}

	return LAYER_ERROR_ERASELAYER_NOTFOUND;
}

/** ���͌����C���[�����擾���� */
unsigned int NNLayer_FeedforwardBase::GetInputFromLayerCount()const
{
	return this->lppInputFromLayer.size();
}
/** ���͌����C���[�̃A�h���X��ԍ��w��Ŏ擾����.
	@param num	�擾���郌�C���[�̔ԍ�.
	@return	���������ꍇ���͌����C���[�̃A�h���X.���s�����ꍇ��NULL���Ԃ�. */
IOutputLayer* NNLayer_FeedforwardBase::GetInputFromLayerByNum(unsigned int num)const
{
	if(num >= this->lppInputFromLayer.size())
		return NULL;

	return this->lppInputFromLayer[num];
}

/** ���͌����C���[�����̓o�b�t�@�̂ǂ̈ʒu�ɋ��邩��Ԃ�.
	���Ώۓ��̓��C���[�̑O�ɂ����̓��̓o�b�t�@�����݂��邩.
	�@�w�K�����̎g�p�J�n�ʒu�Ƃ��Ă��g�p����.
	@return ���s�����ꍇ���̒l���Ԃ� */
int NNLayer_FeedforwardBase::GetInputBufferPositionByLayer(const IOutputLayer* pLayer)
{
	unsigned int bufferPos = 0;

	for(unsigned int layerNum=0; layerNum<this->lppInputFromLayer.size(); layerNum++)
	{
		if(this->lppInputFromLayer[layerNum] == pLayer)
			return bufferPos;

		bufferPos += this->lppInputFromLayer[layerNum]->GetOutputBufferCount();
	}

	return -1;
}


//===========================
// �o�̓��C���[�֘A
//===========================
/** �o�̓f�[�^�\�����擾���� */
IODataStruct NNLayer_FeedforwardBase::GetOutputDataStruct()const
{
	if(this->pConfig == NULL)
		return IODataStruct();

	IODataStruct outputDataStruct;

	outputDataStruct.x = 1;
	outputDataStruct.y = 1;
	outputDataStruct.z = 1;
	outputDataStruct.t = 1;
	outputDataStruct.ch = this->GetNeuronCount();

	return outputDataStruct;
}

/** �o�̓o�b�t�@�����擾���� */
unsigned int NNLayer_FeedforwardBase::GetOutputBufferCount()const
{
	IODataStruct outputDataStruct = GetOutputDataStruct();

	return outputDataStruct.ch * outputDataStruct.x * outputDataStruct.y * outputDataStruct.z * outputDataStruct.t;
}

/** �o�͐惌�C���[�ւ̃����N��ǉ�����.
	@param	pLayer	�ǉ�����o�͐惌�C���[
	@return	���������ꍇ0 */
ELayerErrorCode NNLayer_FeedforwardBase::AddOutputToLayer(class IInputLayer* pLayer)
{
	// �����o�͐惌�C���[�����݂��Ȃ��m�F����
	for(auto it : this->lppOutputToLayer)
	{
		if(it == pLayer)
			return LAYER_ERROR_ADDLAYER_ALREADY_SAMEID;
	}


	// ���X�g�ɒǉ�
	this->lppOutputToLayer.push_back(pLayer);

	// �o�͐惌�C���[�Ɏ�������͌��Ƃ��Ēǉ�
	pLayer->AddInputFromLayer(this);

	return LAYER_ERROR_NONE;
}
/** �o�͐惌�C���[�ւ̃����N���폜����.
	@param	pLayer	�폜����o�͐惌�C���[ */
ELayerErrorCode NNLayer_FeedforwardBase::EraseOutputToLayer(class IInputLayer* pLayer)
{
	// ���X�g���猟�����č폜
	auto it = this->lppOutputToLayer.begin();
	while(it != this->lppOutputToLayer.end())
	{
		if(*it == pLayer)
		{
			// ���X�g����폜
			this->lppOutputToLayer.erase(it);

			// �폜���C���[�ɓo�^����Ă��鎩�����g���폜
			pLayer->EraseInputFromLayer(this);

			return LAYER_ERROR_NONE;
		}
		it++;
	}

	return LAYER_ERROR_ERASELAYER_NOTFOUND;
}

/** �o�͐惌�C���[�����擾���� */
unsigned int NNLayer_FeedforwardBase::GetOutputToLayerCount()const
{
	return this->lppOutputToLayer.size();
}
/** �o�͐惌�C���[�̃A�h���X��ԍ��w��Ŏ擾����.
	@param num	�擾���郌�C���[�̔ԍ�.
	@return	���������ꍇ�o�͐惌�C���[�̃A�h���X.���s�����ꍇ��NULL���Ԃ�. */
IInputLayer* NNLayer_FeedforwardBase::GetOutputToLayerByNum(unsigned int num)const
{
	if(num > this->lppOutputToLayer.size())
		return NULL;

	return this->lppOutputToLayer[num];
}

//===========================
// �ŗL�֐�
//===========================
/** �j���[���������擾���� */
unsigned int NNLayer_FeedforwardBase::GetNeuronCount()const
{
	if(pConfig == NULL)
		return 0;

	const INNLayerConfigItem_Int* pConfigItem = (const INNLayerConfigItem_Int*)pConfig->GetItemByNum(0);
	if(pConfigItem == NULL)
		return 0;

	return (unsigned int)pConfigItem->GetValue();
}