//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
//======================================
#include"stdafx.h"

#include<boost/uuid/uuid_generators.hpp>

#include"FeedforwardNeuralNetwork_Base.h"
#include"FeedforwardNeuralNetwork_LayerData_Base.h"

#include"Layer/NeuralNetwork/INNSingle2SingleLayer.h"
#include"Layer/NeuralNetwork/INNSingle2MultLayer.h"
#include"Layer/NeuralNetwork/INNMult2SingleLayer.h"

#include"LayerConnectInput.h"
#include"LayerConnectOutput.h"
#include"LayerConnectSingle2Single.h"
#include"LayerConnectSingle2Mult.h"
#include"LayerConnectMult2Single.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	//====================================
	// �R���X�g���N�^/�f�X�g���N�^
	//====================================
	/** �R���X�g���N�^ */
	FeedforwardNeuralNetwork_Base::FeedforwardNeuralNetwork_Base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData)
		:	layerData			(i_layerData)
		,	guid				(i_guid)			/**< ���C���[���ʗp��GUID */
		,	inputLayer			(*this)	/**< ���͐M���̑�փ��C���[�̃A�h���X. */
		,	outputLayer			(*this)	/**< �o�͐M���̑�փ��C���[�̃A�h���X. */
		,	pLearnData			(NULL)
		,	m_lppDInputBuffer	(NULL)		/**< ���͌덷�o�b�t�@ */
	{

	}
	/** �f�X�g���N�^ */
	FeedforwardNeuralNetwork_Base::~FeedforwardNeuralNetwork_Base()
	{
		// ���C���[����������`�̍폜
		this->lpCalculateLayerList.clear();

		// ���o�͑�փ��C���[�̍폜

		// ���C���[�ڑ����̍폜
		{
			auto it = this->lpLayerInfo.begin();
			while(it != this->lpLayerInfo.end())
			{
				if(it->second)
					delete it->second;
				it = this->lpLayerInfo.erase(it);
			}
		}

		// �ꎞ���C���[�̃��C���[�f�[�^���폜
		for(U32 i=0; i<this->lpTemporaryLayerData.size(); i++)
			delete this->lpTemporaryLayerData[i];
		this->lpTemporaryLayerData.clear();

		// �w�K�f�[�^�̍폜
		if(this->pLearnData)
			delete this->pLearnData;
	}


	//====================================
	// ���C���[�̒ǉ�
	//====================================
	/** ���C���[��ǉ�����.
		�ǉ��������C���[�̏��L����NeuralNetwork�Ɉڂ邽�߁A�������̊J�������Ȃǂ͑S��INeuralNetwork���ōs����.
		@param pLayer	�ǉ����郌�C���[�̃A�h���X. */
	ErrorCode FeedforwardNeuralNetwork_Base::AddLayer(ILayerBase* pLayer)
	{
		// NULL�`�F�b�N
		if(pLayer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		// �������C���[���o�^�ς݂łȂ����Ƃ��m�F
		if(this->lpLayerInfo.count(pLayer->GetGUID()))
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// ���C���[�̓��o�͎�ʂɉ����ĕ���
		if((pLayer->GetLayerKind() & Gravisbell::Layer::LAYER_KIND_USETYPE) == LAYER_KIND_NEURALNETWORK)
		{
			INNSingle2SingleLayer*  pSingle2SingleLayer  = dynamic_cast<INNSingle2SingleLayer*>(pLayer);
			INNSingle2MultLayer*    pSingle2MultLayer    = dynamic_cast<INNSingle2MultLayer*>(pLayer);
			INNMult2SingleLayer*    pMult2SingleLayer    = dynamic_cast<INNMult2SingleLayer*>(pLayer);

			if(pSingle2SingleLayer)
			{
				// �P�����, �P��o��
				this->lpLayerInfo[pLayer->GetGUID()] = new LayerConnectSingle2Single(*this, pLayer, this->layerData.GetLayerDLLManager().GetLayerDLLByGUID(pLayer->GetLayerCode())->CreateLearningSetting());
			}
			else if(pSingle2MultLayer)
			{
				// �P�����, �����o��
				this->lpLayerInfo[pLayer->GetGUID()] = new LayerConnectSingle2Mult(*this, pLayer, this->layerData.GetLayerDLLManager().GetLayerDLLByGUID(pLayer->GetLayerCode())->CreateLearningSetting());
			}
			else if(pMult2SingleLayer)
			{
				// ��������, �P��o��
				this->lpLayerInfo[pLayer->GetGUID()] = new LayerConnectMult2Single(*this, pLayer, this->layerData.GetLayerDLLManager().GetLayerDLLByGUID(pLayer->GetLayerCode())->CreateLearningSetting());
			}
			else
			{
				// ���Ή�
				return ErrorCode::ERROR_CODE_ADDLAYER_NOT_COMPATIBLE;
			}
		}
		else
		{
			// ���Ή�
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_COMPATIBLE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �ꎞ���C���[��ǉ�����.
			�ǉ��������C���[�f�[�^�̏��L����NeuralNetwork�Ɉڂ邽�߁A�������̊J�������Ȃǂ͑S��INeuralNetwork���ōs����.
			@param	i_pLayerData	�ǉ����郌�C���[�f�[�^.
			@param	o_player		�ǉ����ꂽ���C���[�̃A�h���X. */
	ErrorCode FeedforwardNeuralNetwork_Base::AddTemporaryLayer(ILayerData* i_pLayerData, ILayerBase** o_pLayer)
	{
		if(i_pLayerData == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		lpTemporaryLayerData.push_back(i_pLayerData);

		ILayerBase* pLayer = i_pLayerData->CreateLayer(boost::uuids::random_generator()().data);
		if(pLayer == NULL)
			return ErrorCode::ERROR_CODE_LAYER_CREATE;

		ErrorCode err = this->AddLayer(pLayer);
		if(err != ErrorCode::ERROR_CODE_NONE)
		{
			delete pLayer;
			return err;
		}
		*o_pLayer = pLayer;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[���폜����.
		@param i_guid	�폜���郌�C���[��GUID */
	ErrorCode FeedforwardNeuralNetwork_Base::EraseLayer(const Gravisbell::GUID& i_guid)
	{
		auto it = this->lpLayerInfo.find(i_guid);
		if(it == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		if(it->second)
		{
			// �ڑ�����
			it->second->Disconnect();

			// �{�̍폜
			delete it->second;
		}
		
		// �̈�폜
		this->lpLayerInfo.erase(it);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[��S�폜���� */
	ErrorCode FeedforwardNeuralNetwork_Base::EraseAllLayer()
	{
		auto it = this->lpLayerInfo.begin();
		while(it != this->lpLayerInfo.end())
		{
			if(it->second)
				delete it->second;
			it = this->lpLayerInfo.erase(it);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�^����Ă��郌�C���[�����擾���� */
	U32 FeedforwardNeuralNetwork_Base::GetLayerCount()const
	{
		return (U32)this->lpLayerInfo.size();
	}
	/** ���C���[��GUID��ԍ��w��Ŏ擾���� */
	ErrorCode FeedforwardNeuralNetwork_Base::GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid)
	{
		// �͈̓`�F�b�N
		if(i_layerNum >= this->lpLayerInfo.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// �C�e���[�^��i�߂�
		auto it = this->lpLayerInfo.begin();
		for(U32 i=0; i<i_layerNum; i++)
			it++;

		o_guid = it->first;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//====================================
	// ���o�̓��C���[
	//====================================
	/** ���͐M���Ɋ��蓖�Ă��Ă���GUID���擾���� */
	GUID FeedforwardNeuralNetwork_Base::GetInputGUID()const
	{
		return this->layerData.GetInputGUID();
	}

	/** �o�͐M�����C���[��ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_Base::SetOutputLayerGUID(const Gravisbell::GUID& i_guid)
	{
		// �w�背�C���[�����݂��邱�Ƃ��m�F����
		auto it_layer = this->lpLayerInfo.find(i_guid);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// ���݂̏o�̓��C���[���폜����
		this->outputLayer.Disconnect();

		// �w�背�C���[���o�̓��C���[�ɐڑ�����
		return this->outputLayer.AddInputLayerToLayer(it_layer->second);
	}


	//====================================
	// ���C���[�̐ڑ�
	//====================================
	/** ���C���[�ɓ��̓��C���[��ǉ�����.
		@param	receiveLayer	���͂��󂯎�郌�C���[
		@param	postLayer		���͂�n��(�o�͂���)���C���[. */
	ErrorCode FeedforwardNeuralNetwork_Base::AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		// �󂯎�葤���C���[�����݂��邱�Ƃ��m�F
		auto it_receive = this->lpLayerInfo.find(receiveLayer);
		if(it_receive == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		if(postLayer == this->GetInputGUID())
		{
			// ���̓��C���[�̏ꍇ
			return it_receive->second->AddInputLayerToLayer(&this->inputLayer);
		}
		else
		{
			// �o�͑����C���[�����݂��邱�Ƃ��m�F
			auto it_post = this->lpLayerInfo.find(postLayer);
			if(it_post == this->lpLayerInfo.end())
				return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

			// �ǉ�����
			return it_receive->second->AddInputLayerToLayer(it_post->second);
		}
	}
	/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.
		@param	receiveLayer	���͂��󂯎�郌�C���[
		@param	postLayer		���͂�n��(�o�͂���)���C���[. */
	ErrorCode FeedforwardNeuralNetwork_Base::AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		// �󂯎�葤���C���[�����݂��邱�Ƃ��m�F
		auto it_receive = this->lpLayerInfo.find(receiveLayer);
		if(it_receive == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		if(postLayer == this->GetInputGUID())
		{
			// ���̓��C���[�̏ꍇ
			return it_receive->second->AddBypassLayerToLayer(&this->inputLayer);
		}
		else
		{
			// �o�͑����C���[�����݂��邱�Ƃ��m�F
			auto it_post = this->lpLayerInfo.find(postLayer);
			if(it_post == this->lpLayerInfo.end())
				return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

			// �ǉ�����
			return it_receive->second->AddBypassLayerToLayer(it_post->second);
		}
	}

	/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode FeedforwardNeuralNetwork_Base::ResetInputLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// �w�背�C���[�����݂��邱�Ƃ��m�F����
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		return it_layer->second->ResetInputLayer();
	}
	/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode FeedforwardNeuralNetwork_Base::ResetBypassLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// �w�背�C���[�����݂��邱�Ƃ��m�F����
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		return it_layer->second->ResetBypassLayer();
	}

	/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 FeedforwardNeuralNetwork_Base::GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)const
	{
		// �w�背�C���[�����݂��邱�Ƃ��m�F����
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		return it_layer->second->GetInputLayerCount();
	}
	/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
		@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
	ErrorCode FeedforwardNeuralNetwork_Base::GetInputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const
	{
		// �w�背�C���[�����݂��邱�Ƃ��m�F����
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		// ���̓��C���[�̐ڑ������擾����
		auto pInputLayer = it_layer->second->GetInputLayerByNum(i_inputNum);
		if(pInputLayer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// GUID���R�s�[
		o_postLayerGUID = pInputLayer->GetGUID();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 FeedforwardNeuralNetwork_Base::GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)const
	{
		// �w�背�C���[�����݂��邱�Ƃ��m�F����
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		return it_layer->second->GetBypassLayerCount();
	}
	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
		@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
	ErrorCode FeedforwardNeuralNetwork_Base::GetBypassLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const
	{
		// �w�背�C���[�����݂��邱�Ƃ��m�F����
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		// ���̓��C���[�̐ڑ������擾����
		auto pInputLayer = it_layer->second->GetBypassLayerByNum(i_inputNum);
		if(pInputLayer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// GUID���R�s�[
		o_postLayerGUID = pInputLayer->GetGUID();

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���C���[�̐ڑ���ԂɈُ킪�Ȃ����`�F�b�N����.
		@param	o_errorLayer	�G���[�������������C���[GUID�i�[��. 
		@return	�ڑ��Ɉُ킪�Ȃ��ꍇ��NO_ERROR, �ُ킪�������ꍇ�ُ͈���e��Ԃ��A�Ώۃ��C���[��GUID��o_errorLayer�Ɋi�[����. */
	ErrorCode FeedforwardNeuralNetwork_Base::CheckAllConnection(Gravisbell::GUID& o_errorLayer)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	//====================================
	// �w�K�ݒ�
	//====================================
	/** �w�K�ݒ���擾����.
		�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
		@param	guid	�擾�Ώۃ��C���[��GUID. */
	const SettingData::Standard::IData* FeedforwardNeuralNetwork_Base::GetLearnSettingData(const Gravisbell::GUID& guid)const
	{
		// �w�背�C���[�����݂��邱�Ƃ��m�F����
		auto it_layer = this->lpLayerInfo.find(guid);
		if(it_layer == this->lpLayerInfo.end())
			return NULL;

		return it_layer->second->GetLearnSettingData();
	}
	SettingData::Standard::IData* FeedforwardNeuralNetwork_Base::GetLearnSettingData(const Gravisbell::GUID& guid)
	{
		return const_cast<SettingData::Standard::IData*>( ((const FeedforwardNeuralNetwork_Base*)this)->GetLearnSettingData(guid) );
	}
	
	/** �w�K�ݒ�̃A�C�e�����擾����.
		@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
		@param	i_dataID	�ݒ肷��l��ID. */
	SettingData::Standard::IItemBase* FeedforwardNeuralNetwork_Base::GetLearnSettingDataItem(const Gravisbell::GUID& guid, const wchar_t* i_dataID)
	{
		// �w�K�ݒ�f�[�^���擾
		Gravisbell::SettingData::Standard::IData* pLearnSettingData = this->GetLearnSettingData(guid);
		if(pLearnSettingData == NULL)
			return NULL;

		// �Y��ID�̐ݒ�A�C�e�����擾
		Gravisbell::SettingData::Standard::IItemBase* pItem = pLearnSettingData->GetItemByID(i_dataID);
		if(pItem == NULL)
			return NULL;

		return pItem;
	}

	/** �w�K�ݒ��ݒ肷��.
		�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
		int�^�Afloat�^�Aenum�^���Ώ�.
		@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
		@param	i_dataID	�ݒ肷��l��ID.
		@param	i_param		�ݒ肷��l. */
	ErrorCode FeedforwardNeuralNetwork_Base::SetLearnSettingData(const wchar_t* i_dataID, S32 i_param)
	{
		for(auto& it : this->lpLayerInfo)
			this->SetLearnSettingData(it.first, i_dataID, i_param);
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_Base::SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param)
	{
		// �Y��ID�̐ݒ�A�C�e�����擾
		Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetLearnSettingDataItem(guid, i_dataID);
		if(pItem == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		switch(pItem->GetItemType())
		{
		case Gravisbell::SettingData::Standard::ITEMTYPE_INT:
			{
				Gravisbell::SettingData::Standard::IItem_Int* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Int*>(pItem);
				if(pItemInt == NULL)
					break;
				pItemInt->SetValue(i_param);

				return ErrorCode::ERROR_CODE_NONE;
			}
			break;
		case Gravisbell::SettingData::Standard::ITEMTYPE_FLOAT:
			{
				Gravisbell::SettingData::Standard::IItem_Float* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pItem);
				if(pItemInt == NULL)
					break;
				pItemInt->SetValue((F32)i_param);

				return ErrorCode::ERROR_CODE_NONE;
			}
			break;
		case Gravisbell::SettingData::Standard::ITEMTYPE_ENUM:
			{
				Gravisbell::SettingData::Standard::IItem_Enum* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Enum*>(pItem);
				if(pItemInt == NULL)
					break;
				pItemInt->SetValue(i_param);

				return ErrorCode::ERROR_CODE_NONE;
			}
			break;
		}

		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** �w�K�ݒ��ݒ肷��.
		�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
		int�^�Afloat�^���Ώ�.
		@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
		@param	i_dataID	�ݒ肷��l��ID.
		@param	i_param		�ݒ肷��l. */
	ErrorCode FeedforwardNeuralNetwork_Base::SetLearnSettingData(const wchar_t* i_dataID, F32 i_param)
	{
		for(auto& it : this->lpLayerInfo)
			this->SetLearnSettingData(it.first, i_dataID, i_param);
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_Base::SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param)
	{
		// �Y��ID�̐ݒ�A�C�e�����擾
		Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetLearnSettingDataItem(guid, i_dataID);
		if(pItem == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		switch(pItem->GetItemType())
		{
		case Gravisbell::SettingData::Standard::ITEMTYPE_INT:
			{
				Gravisbell::SettingData::Standard::IItem_Int* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Int*>(pItem);
				if(pItemInt == NULL)
					break;
				pItemInt->SetValue((S32)i_param);

				return ErrorCode::ERROR_CODE_NONE;
			}
			break;
		case Gravisbell::SettingData::Standard::ITEMTYPE_FLOAT:
			{
				Gravisbell::SettingData::Standard::IItem_Float* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pItem);
				if(pItemInt == NULL)
					break;
				pItemInt->SetValue((F32)i_param);

				return ErrorCode::ERROR_CODE_NONE;
			}
			break;
		}

		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** �w�K�ݒ��ݒ肷��.
		�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
		bool�^���Ώ�.
		@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
		@param	i_dataID	�ݒ肷��l��ID.
		@param	i_param		�ݒ肷��l. */
	ErrorCode FeedforwardNeuralNetwork_Base::SetLearnSettingData(const wchar_t* i_dataID, bool i_param)
	{
		for(auto& it : this->lpLayerInfo)
			this->SetLearnSettingData(it.first, i_dataID, i_param);
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_Base::SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param)
	{
		// �Y��ID�̐ݒ�A�C�e�����擾
		Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetLearnSettingDataItem(guid, i_dataID);
		if(pItem == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		switch(pItem->GetItemType())
		{
		case Gravisbell::SettingData::Standard::ITEMTYPE_BOOL:
			{
				Gravisbell::SettingData::Standard::IItem_Bool* pItemBool = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Bool*>(pItem);
				if(pItemBool == NULL)
					break;
				pItemBool->SetValue(i_param);

				return ErrorCode::ERROR_CODE_NONE;
			}
			break;
		}

		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** �w�K�ݒ��ݒ肷��.
		�ݒ肵���l��PreProcessLearnLoop���Ăяo�����ۂɓK�p�����.
		string�^���Ώ�.
		@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
		@param	i_dataID	�ݒ肷��l��ID.
		@param	i_param		�ݒ肷��l. */
	ErrorCode FeedforwardNeuralNetwork_Base::SetLearnSettingData(const wchar_t* i_dataID, const wchar_t* i_param)
	{
		for(auto& it : this->lpLayerInfo)
			this->SetLearnSettingData(it.first, i_dataID, i_param);
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_Base::SetLearnSettingData(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param)
	{
		// �Y��ID�̐ݒ�A�C�e�����擾
		Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetLearnSettingDataItem(guid, i_dataID);
		if(pItem == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		switch(pItem->GetItemType())
		{
		case Gravisbell::SettingData::Standard::ITEMTYPE_STRING:
			{
				Gravisbell::SettingData::Standard::IItem_String* pItemString = dynamic_cast<Gravisbell::SettingData::Standard::IItem_String*>(pItem);
				if(pItemString == NULL)
					break;
				pItemString->SetValue(i_param);

				return ErrorCode::ERROR_CODE_NONE;
			}
			break;
		case Gravisbell::SettingData::Standard::ITEMTYPE_ENUM:
			{
				Gravisbell::SettingData::Standard::IItem_Enum* pItemString = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Enum*>(pItem);
				if(pItemString == NULL)
					break;
				pItemString->SetValue(i_param);

				return ErrorCode::ERROR_CODE_NONE;
			}
			break;
		}

		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}


	//====================================
	// ���o�̓o�b�t�@�֘A
	//====================================

	/** ���̓o�b�t�@���擾���� */
	CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_Base::GetInputBuffer()const
	{
		return this->m_lppInputBuffer;
	}
	/** �o�͍����o�b�t�@���擾���� */
	CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_Base::GetDOutputBuffer()const
	{
		return this->m_lppDOutputBuffer;
	}
		
	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_Base::GetOutputBuffer()const
	{
		return this->outputLayer.GetOutputBuffer();
	}

	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_Base::GetDInputBuffer()
	{
		return this->m_lppDInputBuffer;
	}
	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_Base::GetDInputBuffer()const
	{
		return this->m_lppDInputBuffer;
	}


	//===========================
	// ���C���[����
	//===========================
	/** ���C���[��ʂ̎擾.
		ELayerKind �̑g�ݍ��킹. */
	U32 FeedforwardNeuralNetwork_Base::GetLayerKindBase(void)const
	{
		return Gravisbell::Layer::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::LAYER_KIND_SINGLE_OUTPUT | Gravisbell::Layer::LAYER_KIND_NEURALNETWORK;
	}

	/** ���C���[�ŗL��GUID���擾���� */
	Gravisbell::GUID FeedforwardNeuralNetwork_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** ���C���[�̎�ގ��ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	Gravisbell::GUID FeedforwardNeuralNetwork_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID guid;
		::GetLayerCode(guid);

		return guid;
	}

	/** �o�b�`�T�C�Y���擾����.
		@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
	U32 FeedforwardNeuralNetwork_Base::GetBatchSize()const
	{
		return this->batchSize;
	}


	//================================
	// ����������
	//================================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode FeedforwardNeuralNetwork_Base::Initialize(void)
	{
		for(auto& it : this->lpLayerInfo)
		{
			ErrorCode err = it.second->Initialize();
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}
		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ݒ�
	//===========================
	/** ���C���[�̐ݒ�����擾���� */
	const SettingData::Standard::IData* FeedforwardNeuralNetwork_Base::GetLayerStructure()const
	{
		return this->layerData.GetLayerStructure();
	}


	//===========================
	// ���̓��C���[�֘A
	//===========================
	/** ���̓f�[�^�\�����擾����.
		@return	���̓f�[�^�\�� */
	IODataStruct FeedforwardNeuralNetwork_Base::GetInputDataStruct()const
	{
		return this->layerData.GetInputDataStruct();
	}

	/** ���̓o�b�t�@�����擾����. */
	U32 FeedforwardNeuralNetwork_Base::GetInputBufferCount()const
	{
		return this->layerData.GetInputBufferCount();
	}


	//===========================
	// �o�̓��C���[�֘A
	//===========================
	/** �o�̓f�[�^�\�����擾���� */
	IODataStruct FeedforwardNeuralNetwork_Base::GetOutputDataStruct()const
	{
		return this->outputLayer.GetOutputDataStruct();
	}

	/** �o�̓o�b�t�@�����擾���� */
	U32 FeedforwardNeuralNetwork_Base::GetOutputBufferCount()const
	{
		return this->outputLayer.GetOutputDataStruct().GetDataCount();
	}


	//================================
	// ���Z����
	//================================
	/** �ڑ��̊m�����s�� */
	ErrorCode FeedforwardNeuralNetwork_Base::EstablishmentConnection(void)
	{
		ErrorCode err = ErrorCode::ERROR_CODE_NONE;

		// �ڑ����X�g���N���A
		this->lpCalculateLayerList.clear();

		// ���C���[��GUID���X�g�𐶐�
		std::set<Gravisbell::GUID> lpLayerGUID;	// �S���C���[���X�g
		err = this->outputLayer.CreateLayerList(lpLayerGUID);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ���������X�g�̍쐬
		std::set<Gravisbell::GUID> lpAddedList;	// �ǉ��σ��C���[���X�g
		std::set<ILayerConnect*> lpAddWaitList;	// �ǉ��ҋ@��ԃ��X�g
		// �ŏ���1��ڂ����s
		err = this->outputLayer.CreateCalculateList(lpLayerGUID, lpCalculateLayerList, lpAddedList, lpAddWaitList);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;
		// �ҋ@��Ԃ̃��C���[�����������s����
		while(lpAddWaitList.size())
		{
			bool onAddElse = false;	// ��ȏ�̃��C���[��ǉ����邱�Ƃ��ł���
			for(auto it : lpAddWaitList)
			{
				bool onHaveNonAddedOutputLayer = false;	// �ǉ��ςłȂ����C���[���o�͑Ώۂɕێ����Ă���t���O
				for(U32 outputLayerNum=0; outputLayerNum<it->GetOutputToLayerCount(); outputLayerNum++)
				{
					auto pOutputToLayer = it->GetOutputToLayerByNum(outputLayerNum);

					// ���C���[�ꗗ�Ɋ܂܂�Ă��邱�Ƃ��m�F
					if(lpLayerGUID.count(pOutputToLayer->GetGUID()) == 0)
						continue;

					// �ǉ��ҋ@��ԂɊ܂܂�Ă��Ȃ����Ƃ��m�F
					if(lpAddWaitList.count(pOutputToLayer) == 0)
						continue;

					onHaveNonAddedOutputLayer = true;
					break;
				}

				if(onHaveNonAddedOutputLayer == false)
				{
					// �ǉ��ҋ@��ԂɊ܂܂�Ă���o�̓��C���[�����݂��Ȃ����߁A���Z���X�g�ɒǉ��\

					// �ǉ��ҋ@����
					lpAddWaitList.erase(it);

					// ���Z���X�g�ɒǉ�
					it->CreateCalculateList(lpLayerGUID, this->lpCalculateLayerList, lpAddedList, lpAddWaitList);

					onAddElse = true;
					break;
				}
			}

			if(!onAddElse)
			{
				// �����ꂩ�̃��C���[��ǉ����邱�Ƃ��ł��Ȃ����� = �ċN�����ɂȂ��Ă��邽�߁A�G���[
				return ERROR_CODE_COMMON_NOT_COMPATIBLE;
			}
		}


		// �ڑ��̊m��
		for(auto& it : this->lpCalculateLayerList)
		{
			err = it->EstablishmentConnection();
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode FeedforwardNeuralNetwork_Base::PreProcessLearn(U32 batchSize)
	{
		ErrorCode err = ErrorCode::ERROR_CODE_NONE;

		// �o�b�`�T�C�Y���i�[����
		this->batchSize = batchSize;

		// �ڑ��̊m�����s��
		err = this->EstablishmentConnection();
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;


		// ���C���[���g�p������͌덷�o�b�t�@�����蓖�Ă�
		{
			struct DInputBufferInfo
			{
				U32							maxBufferSize;	/**< �ő�o�b�t�@�T�C�Y */
				std::set<Gravisbell::GUID>	lpUseLayerID;	/**< �g�p���̃��C���[ID */

				DInputBufferInfo()
					:	maxBufferSize	(0)
					,	lpUseLayerID	()
				{
				}
			};

			std::map<U32, DInputBufferInfo> lpDInputBufferInfo;	/**< ���͌덷�o�b�t�@�̎g�p��<���͌덷�o�b�t�@��ID, �g�p���̃��C���[��GUID>  */

			auto it_layer = this->lpCalculateLayerList.rbegin();
			while(it_layer != this->lpCalculateLayerList.rend())
			{
				// ���̓��C���[�𑖍����ē��͌덷�o�b�t�@��ID�����蓖�Ă�
				for(U32 inputNum=0; inputNum<(*it_layer)->GetInputLayerCount(); inputNum++)
				{
					auto pInputLayer = (*it_layer)->GetInputLayerByNum(inputNum);
					if(pInputLayer->GetGUID() == this->GetInputGUID())
					{
						// ���̓��C���[
						(*it_layer)->SetDInputBufferID(inputNum, -1);
					}
					else if(pInputLayer == NULL)
					{
						// ���͂����蓖�Ă��Ă��Ȃ��̂ł��肦�Ȃ��o�b�t�@ID��ݒ肷��
						(*it_layer)->SetDInputBufferID(inputNum, 0xFFFF);
					}
					else
					{
						// �ʏ탌�C���[

						// ���g�p�̓��͌덷�o�b�t�@������
						S32 useDInputBufferID = -1;
						for(auto& it_DInputBuffer : lpDInputBufferInfo)
						{
							if(it_DInputBuffer.second.lpUseLayerID.size() == 0)
							{
								useDInputBufferID = (S32)it_DInputBuffer.first;
								break;
							}
						}
						if(useDInputBufferID < 0)
							useDInputBufferID = (S32)lpDInputBufferInfo.size();

						// ���͌덷�o�b�t�@���g�p���ɕύX���A�ő�o�b�t�@�T�C�Y���X�V
						lpDInputBufferInfo[(U32)useDInputBufferID].lpUseLayerID.insert(pInputLayer->GetGUID());
						lpDInputBufferInfo[(U32)useDInputBufferID].maxBufferSize = max(lpDInputBufferInfo[(U32)useDInputBufferID].maxBufferSize, pInputLayer->GetOutputDataStruct().GetDataCount());

						// ���͌덷�o�b�t�@��ID��o�^
						(*it_layer)->SetDInputBufferID(inputNum, useDInputBufferID);

						// �������g�p���Ă�����͌덷�o�b�t�@���J��
						for(auto& it_DInputBuffer : lpDInputBufferInfo)
						{
							if(it_DInputBuffer.second.lpUseLayerID.count((*it_layer)->GetGUID()) > 0)
							{
								it_DInputBuffer.second.lpUseLayerID.erase((*it_layer)->GetGUID());
							}
						}
					}
				}

				it_layer++;
			}

			// ���͌덷�o�b�t�@���m�ۂ���
			this->SetDInputBufferCount((U32)lpDInputBufferInfo.size());
			for(U32 dInputBufferNum=0; dInputBufferNum<lpDInputBufferInfo.size(); dInputBufferNum++)
			{
				this->ResizeDInputBuffer(dInputBufferNum, lpDInputBufferInfo[dInputBufferNum].maxBufferSize * this->batchSize);
			}
		}


		// �w�K�̎��O���������s
		auto it = this->lpCalculateLayerList.begin();
		while(it != this->lpCalculateLayerList.end())
		{
			err = (*it)->PreProcessLearn(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			it++;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FeedforwardNeuralNetwork_Base::PreProcessCalculate(unsigned int batchSize)
	{
		ErrorCode err = ErrorCode::ERROR_CODE_NONE;

		// �o�b�`�T�C�Y���i�[����
		this->batchSize = batchSize;

		// �ڑ��̊m�����s��
		err = this->EstablishmentConnection();
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ���Z�̎��O���������s
		auto it = this->lpCalculateLayerList.begin();
		while(it != this->lpCalculateLayerList.end())
		{
			ErrorCode err = (*it)->PreProcessCalculate(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			it++;
		}
		
		return ErrorCode::ERROR_CODE_NONE;
	}
		
	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FeedforwardNeuralNetwork_Base::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		auto it = this->lpCalculateLayerList.begin();
		while(it != this->lpCalculateLayerList.end())
		{
			ErrorCode err = (*it)->PreProcessLearnLoop();
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			it++;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FeedforwardNeuralNetwork_Base::PreProcessCalculateLoop()
	{
		auto it = this->lpCalculateLayerList.begin();
		while(it != this->lpCalculateLayerList.end())
		{
			ErrorCode err = (*it)->PreProcessCalculateLoop();
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			it++;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode FeedforwardNeuralNetwork_Base::Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		// ���̓o�b�t�@���ꎞ�ۑ�
		this->m_lppInputBuffer = i_lppInputBuffer;

		// ���Z�����s
		auto it = this->lpCalculateLayerList.begin();
		while(it != this->lpCalculateLayerList.end())
		{
			ErrorCode err = (*it)->Calculate();
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			it++;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FeedforwardNeuralNetwork_Base::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�o�b�t�@��ۑ�
		this->m_lppDInputBuffer = o_lppDInputBuffer;

		// �o�͍����o�b�t�@���ꎞ�ۑ�
		this->m_lppDOutputBuffer = i_lppDOutputBuffer;

		// �w�K���������s
		{
			auto it = this->lpCalculateLayerList.rbegin();
			while(it != this->lpCalculateLayerList.rend())
			{
				ErrorCode err = (*it)->CalculateDInput();
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				it++;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FeedforwardNeuralNetwork_Base::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�o�b�t�@��ۑ�
		this->m_lppDInputBuffer = o_lppDInputBuffer;

		// �o�͍����o�b�t�@���ꎞ�ۑ�
		this->m_lppDOutputBuffer = i_lppDOutputBuffer;

		// �w�K���������s
		{
			auto it = this->lpCalculateLayerList.rbegin();
			while(it != this->lpCalculateLayerList.rend())
			{
				ErrorCode err = (*it)->Training();
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				it++;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell