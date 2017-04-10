//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[�̃f�[�^
// �����̃��C���[�����A��������
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_LayerData_Base.h"
#include"FeedforwardNeuralNetwork_FUNC.hpp"
#include"FeedforwardNeuralNetwork_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	//====================================
	// �R���X�g���N�^/�f�X�g���N�^
	//====================================
	/** �R���X�g���N�^ */
	FeedforwardNeuralNetwork_LayerData_Base::FeedforwardNeuralNetwork_LayerData_Base(const ILayerDLLManager& i_layerDLLManager, const Gravisbell::GUID& guid)
		:	layerDLLManager	(i_layerDLLManager)
		,	guid			(guid)
		,	inputLayerGUID	(Gravisbell::GUID(0x2d2805a3, 0x97cc, 0x4ab4, 0x94, 0x2e, 0x69, 0x39, 0xfd, 0x62, 0x35, 0xb1))
	{
	}
	/** �f�X�g���N�^ */
	FeedforwardNeuralNetwork_LayerData_Base::~FeedforwardNeuralNetwork_LayerData_Base()
	{
		// �����ۗL�̃��C���[�f�[�^���폜
		for(auto pLayerData : this->lpLayerData)
			delete pLayerData;

		// ���C���[�\�����폜
		if(this->pLayerStructure)
			delete this->pLayerStructure;
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::Initialize(void)
	{
		for(auto& connectInfo : this->lpConnectInfo)
		{
			connectInfo.second.pLayerData->Initialize();
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
	{
		ErrorCode err;

		// �ݒ���̓o�^
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ���̓f�[�^�\���̐ݒ�
		this->inputDataStruct = i_inputDataStruct;

		return this->Initialize();
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}


	//===========================
	// ���ʐ���
	//===========================
	/** ���C���[�ŗL��GUID���擾���� */
	Gravisbell::GUID FeedforwardNeuralNetwork_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** ���C���[��ʎ��ʃR�[�h���擾����.
		@param o_layerCode	�i�[��o�b�t�@
		@return ���������ꍇ0 */
	Gravisbell::GUID FeedforwardNeuralNetwork_LayerData_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		::GetLayerCode(layerCode);

		return layerCode;
	}

	/** ���C���[DLL�}�l�[�W���̎擾 */
	const ILayerDLLManager& FeedforwardNeuralNetwork_LayerData_Base::GetLayerDLLManager(void)const
	{
		return this->layerDLLManager;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetUseBufferByteCount()const
	{
		// TODO
		return 0;
	}

	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 FeedforwardNeuralNetwork_LayerData_Base::WriteToBuffer(BYTE* o_lpBuffer)const
	{

		//if(this->pLayerStructure == NULL)
		//	return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		//// �ݒ���

		//// ���C���[�̐�

		//// �e���C���[�{��
		//for(auto& it : lpLayerInfo)
		//{
		//	// ���C���[��ʃR�[�h

		//	// ���C���[GUID

		//	// ���C���[�{��
		//}

		//// ���C���[�ڑ����
		//{
		//	// ���C���[�ڑ����ꗗ���쐬
		//	for(auto& it : lpLayerInfo)
		//	{
		//	}

		//	// ���C���[�ڑ����

		//	// ���C���[�ڑ����
		//}

#ifdef _DEBUG
		return -1;
#endif

		// TODO
		return -1;
	}


	//===========================
	// ���C���[�ݒ�
	//===========================
	/** �ݒ����ݒ� */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
	{
		Gravisbell::ErrorCode err = ERROR_CODE_NONE;

		// ���C���[�R�[�h���m�F
		{
			Gravisbell::GUID config_guid;
			err = config.GetLayerCode(config_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			Gravisbell::GUID layer_guid;
			err = ::GetLayerCode(layer_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			if(config_guid != layer_guid)
				return ERROR_CODE_INITLAYER_DISAGREE_CONFIG;
		}

		if(this->pLayerStructure != NULL)
			delete this->pLayerStructure;
		this->pLayerStructure = config.Clone();


		return ERROR_CODE_NONE;
	}
	/** ���C���[�̐ݒ�����擾���� */
	const SettingData::Standard::IData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// ���̓��C���[�֘A
	//===========================
	/** ���̓f�[�^�\�����擾����.
		@return	���̓f�[�^�\�� */
	IODataStruct FeedforwardNeuralNetwork_LayerData_Base::GetInputDataStruct()const
	{
		return this->inputDataStruct;
	}

	/** ���̓o�b�t�@�����擾����. */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetInputBufferCount()const
	{
		return this->inputDataStruct.GetDataCount();
	}


	//===========================
	// �o�̓��C���[�֘A
	//===========================
	/** �o�̓f�[�^�\�����擾���� */
	IODataStruct FeedforwardNeuralNetwork_LayerData_Base::GetOutputDataStruct()const
	{
		auto it = this->lpConnectInfo.find(this->outputLayerGUID);
		if(it == this->lpConnectInfo.end())
			return IODataStruct();

		return it->second.pLayerData->GetOutputDataStruct();
	}

	/** �o�̓o�b�t�@�����擾���� */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetOutputBufferCount()const
	{
		return this->GetOutputDataStruct().GetDataCount();
	}



	//===========================
	// ���C���[�쐬
	//===========================
	/** �쐬���ꂽ�V�K�j���[�����l�b�g���[�N�ɑ΂��ē������C���[��ǉ����� */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddConnectionLayersToNeuralNetwork(class FeedforwardNeuralNetwork_Base& neuralNetwork)
	{
		// �S���C���[��ǉ�����
		for(auto it : this->lpConnectInfo)
		{
			neuralNetwork.AddLayer(it.second.pLayerData->CreateLayer(it.first));
		}

		// ���C���[�Ԃ̐ڑ���ݒ肷��
		for(auto it_connectInfo : this->lpConnectInfo)
		{
			// ���̓��C���[
			for(auto inputGUID : it_connectInfo.second.lpInputLayerGUID)
				neuralNetwork.AddInputLayerToLayer(it_connectInfo.first, inputGUID);

			// �o�C�p�X���C���[
			for(auto bypassGUID : it_connectInfo.second.lpInputLayerGUID)
				neuralNetwork.AddBypassLayerToLayer(it_connectInfo.first, bypassGUID);
		}

		// ���C���[�Ԃ̐ڑ���Ԃ��m�F
		Gravisbell::GUID errorLayerGUID;
		return neuralNetwork.CheckAllConnection(errorLayerGUID);
	}


	//====================================
	// ���C���[�̒ǉ�/�폜/�Ǘ�
	//====================================
	/** ���C���[�f�[�^��ǉ�����.
		@param	i_guid			�ǉ����郌�C���[�Ɋ��蓖�Ă���GUID.
		@param	i_pLayerData	�ǉ����郌�C���[�f�[�^�̃A�h���X. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddLayer(const Gravisbell::GUID& i_guid, INNLayerData* i_pLayerData)
	{
		// ���C���[������
		if(this->lpConnectInfo.count(i_guid) != 0)
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// �ǉ�
		this->lpConnectInfo[i_guid] = LayerConnect(i_guid, i_pLayerData);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[�f�[�^���폜����.
		@param i_guid	�폜���郌�C���[��GUID */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::EraseLayer(const Gravisbell::GUID& i_guid)
	{
		// �폜���C���[������
		auto it = this->lpConnectInfo.find(i_guid);
		if(it == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		// �폜�Ώۂ̃��C���[����͂Ɏ��ꍇ�͍폜
		for(auto& it_search : this->lpConnectInfo)
		{
			it_search.second.lpInputLayerGUID.erase(i_guid);
			it_search.second.lpBypassLayerGUID.erase(i_guid);
		}

		// ���C���[�f�[�^�{�̂����ꍇ�͍폜
		{
			auto it_pLayerData = this->lpLayerData.find(it->second.pLayerData);
			if(it_pLayerData != this->lpLayerData.end())
			{
				delete *it_pLayerData;
				this->lpLayerData.erase(it_pLayerData);
			}
		}

		// �ڑ����폜
		this->lpConnectInfo.erase(it);

		// �o�͑Ώۃ��C���[�̏ꍇ��������
		if(this->outputLayerGUID == i_guid)
			this->outputLayerGUID = Gravisbell::GUID();

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[�f�[�^��S�폜���� */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::EraseAllLayer()
	{
		// �ڑ�����S�폜
		this->lpConnectInfo.clear();

		// ���C���[�f�[�^�{�̂��폜
		for(auto pLayerData : this->lpLayerData)
		{
			delete pLayerData;
		}
		this->lpLayerData.clear();

		// �o�͑Ώۃ��C���[�̏ꍇ��������
		this->outputLayerGUID = Gravisbell::GUID();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�^����Ă��郌�C���[�����擾���� */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetLayerCount()
	{
		return this->lpConnectInfo.size();
	}
	/** ���C���[��GUID��ԍ��w��Ŏ擾���� */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid)
	{
		if(i_layerNum >= this->lpConnectInfo.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		auto it = this->lpConnectInfo.begin();
		for(U32 i=0; i<i_layerNum; i++)
			it++;

		o_guid = it->first;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�^����Ă��郌�C���[�f�[�^��ԍ��w��Ŏ擾���� */
	INNLayerData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerDataByNum(U32 i_layerNum)
	{
		Gravisbell::GUID layerGUID;
		ErrorCode err = this->GetLayerGUIDbyNum(i_layerNum, layerGUID);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return NULL;

		return this->GetLayerDataByGUID(layerGUID);
	}
	/** �o�^����Ă��郌�C���[�f�[�^��GUID�w��Ŏ擾���� */
	INNLayerData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerDataByGUID(const Gravisbell::GUID& i_guid)
	{
		auto it = this->lpConnectInfo.find(i_guid);
		if(it == this->lpConnectInfo.end())
			return NULL;

		return it->second.pLayerData;
	}


	//====================================
	// ���o�̓��C���[
	//====================================
	/** ���͐M���Ɋ��蓖�Ă��Ă���GUID���擾���� */
	GUID FeedforwardNeuralNetwork_LayerData_Base::GetInputGUID()
	{
		return this->inputLayerGUID;
	}

	/** �o�͐M�����C���[��ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetOutputLayerGUID(const Gravisbell::GUID& i_guid)
	{
		if(this->lpConnectInfo.count(i_guid) == 0)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		this->outputLayerGUID = i_guid;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//====================================
	// ���C���[�̐ڑ�
	//====================================
	/** ���C���[�ɓ��̓��C���[��ǉ�����.
		@param	receiveLayer	���͂��󂯎�郌�C���[
		@param	postLayer		���͂�n��(�o�͂���)���C���[. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		// ���C���[�̑��݂��m�F
		auto it_receive = this->lpConnectInfo.find(receiveLayer);
		if(it_receive != this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// ���ꃌ�C���[���ǉ��ςłȂ����Ƃ��m�F
		if(it_receive->second.lpInputLayerGUID.count(postLayer) != 0)
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// ���C���[��ǉ�
		it_receive->second.lpInputLayerGUID.insert(postLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.
		@param	receiveLayer	���͂��󂯎�郌�C���[
		@param	postLayer		���͂�n��(�o�͂���)���C���[. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		// ���C���[�̑��݂��m�F
		auto it_receive = this->lpConnectInfo.find(receiveLayer);
		if(it_receive != this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// ���ꃌ�C���[���ǉ��ςłȂ����Ƃ��m�F
		if(it_receive->second.lpBypassLayerGUID.count(postLayer) != 0)
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// ���C���[��ǉ�
		it_receive->second.lpBypassLayerGUID.insert(postLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::ResetInputLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer != this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// �폜����
		it_layer->second.lpInputLayerGUID.clear();

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::ResetBypassLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer != this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// �폜����
		it_layer->second.lpBypassLayerGUID.clear();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer != this->lpConnectInfo.end())
			return 0;

		return it_layer->second.lpInputLayerGUID.size();
	}
	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer != this->lpConnectInfo.end())
			return 0;

		return it_layer->second.lpBypassLayerGUID.size();
	}
	/** ���C���[�ɐڑ����Ă���o�̓��C���[�̐����擾���� */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetOutputLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer != this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// �Ώۃ��C���[����̓��C���[�Ɏ����C���[���𐔂���
		U32 outputCount = 0;
		for(auto it : this->lpConnectInfo)
		{
			if(it.second.lpInputLayerGUID.count(i_layerGUID) != 0)
				outputCount++;
		}

		return outputCount;
	}

	/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
		@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::GetInputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer != this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// ���̓��C���[�̐����m�F
		if(i_inputNum >= it_layer->second.lpInputLayerGUID.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// �C�e���[�^�i�s
		auto it = it_layer->second.lpInputLayerGUID.begin();
		for(U32 i=0; i<i_inputNum; i++)
			it++;

		o_postLayerGUID = *it;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
		@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::GetBypassLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer != this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// ���̓��C���[�̐����m�F
		if(i_inputNum >= it_layer->second.lpBypassLayerGUID.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// �C�e���[�^�i�s
		auto it = it_layer->second.lpBypassLayerGUID.begin();
		for(U32 i=0; i<i_inputNum; i++)
			it++;

		o_postLayerGUID = *it;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[�ɐڑ����Ă���o�̓��C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
		@param	i_outputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
		@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::GetOutputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_outputNum, Gravisbell::GUID& o_postLayerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer != this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// �Ώۃ��C���[����̓��C���[�Ɏ����C���[���𐔂��Ĕԍ�����v������I��
		U32 outputNum = 0;
		for(auto it : this->lpConnectInfo)
		{
			if(it.second.lpInputLayerGUID.count(i_layerGUID) != 0)
			{
				if(outputNum == i_outputNum)
				{
					o_postLayerGUID = it.first;
					return ErrorCode::ERROR_CODE_NONE;
				}
				outputNum++;
			}
		}

		return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell
