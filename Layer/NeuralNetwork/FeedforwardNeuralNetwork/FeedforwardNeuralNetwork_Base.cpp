//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_Base.h"

#include"LayerConnectInput.h"
#include"LayerConnectOutput.h"
#include"LayerConnectSingle2Single.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	//====================================
	// �R���X�g���N�^/�f�X�g���N�^
	//====================================
	/** �R���X�g���N�^ */
	FeedforwardNeuralNetwork_Base::FeedforwardNeuralNetwork_Base(const ILayerDLLManager& layerDLLManager, const Gravisbell::GUID& i_guid, const Gravisbell::GUID& i_inputLayerGUID)
		:	layerDLLManager	(layerDLLManager)
		,	guid			(i_guid)			/**< ���C���[���ʗp��GUID */
		,	inputLayerGUID	(i_inputLayerGUID)	/**< ���͐M���Ɋ��蓖�Ă��Ă���GUID.���͐M�����C���[�̑�p�Ƃ��Ďg�p����. */
		,	inputLayer		(*this)	/**< ���͐M���̑�փ��C���[�̃A�h���X. */
		,	outputLayer		(*this)	/**< �o�͐M���̑�փ��C���[�̃A�h���X. */
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

		// ���C���[�\���̍폜
		if(this->pLayerStructure)
			delete this->pLayerStructure;

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
		switch(pLayer->GetLayerKind() & (Gravisbell::Layer::LAYER_KIND_IOTYPE | Gravisbell::Layer::LAYER_KIND_USETYPE))
		{
		case LAYER_KIND_SINGLE_INPUT | LAYER_KIND_SINGLE_OUTPUT | LAYER_KIND_NEURALNETWORK:
			{
				INNLayer* pNNLayer = dynamic_cast<INNLayer*>(pLayer);
				if(pNNLayer)
				{
					this->lpLayerInfo[pLayer->GetGUID()] = new LayerConnectSingle2Single(pNNLayer, this->layerDLLManager.GetLayerDLLByGUID(pLayer->GetLayerCode())->CreateLearningSetting());
				}
				else
				{
					// ���Ή�
					return ErrorCode::ERROR_CODE_ADDLAYER_NOT_COMPATIBLE;
				}
			}
			break;
		default:
			// ���Ή�
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_COMPATIBLE;
			break;
		}

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
		return this->lpLayerInfo.size();
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
		return this->inputLayerGUID;
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

		// �o�͑����C���[�����݂��邱�Ƃ��m�F
		auto it_post = this->lpLayerInfo.find(postLayer);
		if(it_post == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// �ǉ�����
		return it_receive->second->AddInputLayerToLayer(it_post->second);
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

		// �o�͑����C���[�����݂��邱�Ƃ��m�F
		auto it_post = this->lpLayerInfo.find(postLayer);
		if(it_post == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// �ǉ�����
		return it_receive->second->AddBypassLayerToLayer(it_post->second);
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
	CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_Base::GetDInputBuffer()const
	{
		return this->inputLayer.GetDInputBufferByNum(0);
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
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode FeedforwardNeuralNetwork_Base::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
	{
		// �w�K�ݒ��ۑ�
		if(this->pLayerStructure)
			delete this->pLayerStructure;
		this->pLayerStructure = i_data.Clone();

		// ���̓f�[�^�\����ۑ�
		this->inputDataStruct = i_inputDataStruct;

		// ���C���[����������`�̍폜
		this->lpCalculateLayerList.clear();

		// ���C���[�ڑ����̍폜
		{
			auto it = this->lpLayerInfo.begin();
			while(it != this->lpLayerInfo.end())
			{
				if(it->second)
				{
					it->second->Disconnect();
					delete it->second;
				}
				it = this->lpLayerInfo.erase(it);
			}
		}

		// ���ʂ̏��������������s
		return this->Initialize();
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode FeedforwardNeuralNetwork_Base::InitializeFromBuffer(BYTE* i_lpBuffer, int i_bufferSize)
	{
	}


	//===========================
	// ���C���[�ݒ�
	//===========================
	/** ���C���[�̐ݒ�����擾���� */
	const SettingData::Standard::IData* FeedforwardNeuralNetwork_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U32 FeedforwardNeuralNetwork_Base::GetUseBufferByteCount()const
	{
	}

	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 FeedforwardNeuralNetwork_Base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		// �ݒ���

		// ���C���[�̐�

		// �e���C���[�{��
		for(auto& it : lpLayerInfo)
		{
			// ���C���[��ʃR�[�h

			// ���C���[GUID

			// ���C���[�{��
		}

		// ���C���[�ڑ����
		{
			// ���C���[�ڑ����ꗗ���쐬
			for(auto& it : lpLayerInfo)
			{
			}

			// ���C���[�ڑ����

			// ���C���[�ڑ����
		}

	}


	//===========================
	// ���̓��C���[�֘A
	//===========================
	/** ���̓f�[�^�\�����擾����.
		@return	���̓f�[�^�\�� */
	IODataStruct FeedforwardNeuralNetwork_Base::GetInputDataStruct()const
	{
		return this->inputDataStruct;
	}

	/** ���̓o�b�t�@�����擾����. */
	U32 FeedforwardNeuralNetwork_Base::GetInputBufferCount()const
	{
		return this->inputDataStruct.GetDataCount();
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
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode FeedforwardNeuralNetwork_Base::PreProcessLearn(U32 batchSize)
	{
		// �ڑ��̊m�����s��

		// �w�K�̎��O���������s

	}
	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculate(unsigned int batchSize)
	{
		// �ڑ��̊m�����s��

		// ���Z�̎��O���������s
	}
		
	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FeedforwardNeuralNetwork_Base::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FeedforwardNeuralNetwork_Base::PreProcessCalculateLoop()
	{
	}

	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode FeedforwardNeuralNetwork_Base::Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		// ���̓o�b�t�@���ꎞ�ۑ�
		this->m_lppInputBuffer = i_lppInputBuffer;

		// ���Z�����s
	}


	//================================
	// �w�K����
	//================================
	/** �w�K�덷���v�Z����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FeedforwardNeuralNetwork_Base::CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// �o�͍����o�b�t�@���ꎞ�ۑ�

		// �w�K�덷�v�Z�����s

	}
	/** �w�K���������C���[�ɔ��f������.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		�o�͌덷�����A���͌덷�����͒��O��CalculateLearnError�̒l���Q�Ƃ���. */
	ErrorCode FeedforwardNeuralNetwork_Base::ReflectionLearnError(void)
	{
		// �w�K�덷���f�����s

	}



}	// NeuralNetwork
}	// Layer
}	// Gravisbell