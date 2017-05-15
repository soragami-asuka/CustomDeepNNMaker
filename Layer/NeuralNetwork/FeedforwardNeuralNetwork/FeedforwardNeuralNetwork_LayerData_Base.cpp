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
		,	pLayerStructure	(NULL)
	{
	}
	/** �f�X�g���N�^ */
	FeedforwardNeuralNetwork_LayerData_Base::~FeedforwardNeuralNetwork_LayerData_Base()
	{
		// �����ۗL�̃��C���[�f�[�^���폜
		for(auto it : this->lpLayerData)
			delete it.second;

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
			connectInfo.pLayerData->Initialize();
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
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize)
	{
		int readBufferByte = 0;

		// ���̓f�[�^�\��
		memcpy(&this->inputDataStruct, &i_lpBuffer[readBufferByte], sizeof(this->inputDataStruct));
		readBufferByte += sizeof(this->inputDataStruct);

		// �ݒ���
		S32 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// �����ۗL�̃��C���[�f�[�^���폜
		for(auto it : this->lpLayerData)
			delete it.second;
		this->lpLayerData.clear();

		// ����������
		this->Initialize();
		
		// ���C���[�̐�
		U32 layerDataCount = 0;
		memcpy(&layerDataCount, &i_lpBuffer[readBufferByte], sizeof(U32));
		readBufferByte += sizeof(U32);

		// �e���C���[�f�[�^
		for(U32 layerDataNum=0; layerDataNum<layerDataCount; layerDataNum++)
		{
			// ���C���[��ʃR�[�h
			Gravisbell::GUID typeCode;
			memcpy(&typeCode, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
			readBufferByte += sizeof(Gravisbell::GUID);

			// ���C���[GUID
			Gravisbell::GUID guid;
			memcpy(&guid, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
			readBufferByte += sizeof(Gravisbell::GUID);

			// ���C���[�{��
			auto pLayerDLL = this->layerDLLManager.GetLayerDLLByGUID(typeCode);
			if(pLayerDLL == NULL)
				return ErrorCode::ERROR_CODE_DLL_NOTFOUND;
			S32 useBufferSize = 0;
			auto pLayerData = pLayerDLL->CreateLayerDataFromBuffer(guid, &i_lpBuffer[readBufferByte], readBufferByte, useBufferSize);
			if(pLayerData == NULL)
				return ErrorCode::ERROR_CODE_LAYER_CREATE;
			readBufferByte += useBufferSize;

			// ���C���[�f�[�^�ꗗ�ɒǉ�
			this->lpLayerData[guid] = pLayerData;
		}

		// ���C���[�ڑ����
		{
			// ���C���[�ڑ����
			U32 connectDataCount = 0;
			memcpy(&connectDataCount, &i_lpBuffer[readBufferByte], sizeof(U32));
			readBufferByte += sizeof(U32);

			// ���C���[�ڑ����
			for(U32 connectDataNum=0; connectDataNum<connectDataCount; connectDataNum++)
			{
				// ���C���[��GUID
				Gravisbell::GUID layerGUID;
				memcpy(&layerGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
				readBufferByte += sizeof(Gravisbell::GUID);

				// ���C���[�f�[�^��GUID
				Gravisbell::GUID layerDataGUID;
				memcpy(&layerDataGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
				readBufferByte += sizeof(Gravisbell::GUID);

				// ���C���[�f�[�^�擾
				auto pLayerData = this->lpLayerData[layerDataGUID];
				if(pLayerData == NULL)
					return ErrorCode::ERROR_CODE_LAYER_CREATE;

				// ���C���[�ڑ������쐬
				LayerConnect layerConnect(layerGUID, pLayerData);

				// ���̓��C���[�̐�
				U32 inputLayerCount = 0;
				memcpy(&inputLayerCount, &i_lpBuffer[readBufferByte], sizeof(U32));
				readBufferByte += sizeof(U32);

				// ���̓��C���[
				for(U32 inputLayerNum=0; inputLayerNum<inputLayerCount; inputLayerNum++)
				{
					Gravisbell::GUID inputGUID;
					memcpy(&inputGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
					readBufferByte += sizeof(Gravisbell::GUID);

					layerConnect.lpInputLayerGUID.push_back(inputGUID);
				}

				// �o�C�p�X���̓��C���[�̐�
				U32 bypassLayerCount = 0;
				memcpy(&bypassLayerCount, &i_lpBuffer[readBufferByte], sizeof(U32));
				readBufferByte += sizeof(U32);

				// �o�C�p�X���̓��C���[
				for(U32 bypassLayerNum=0; bypassLayerNum<bypassLayerCount; bypassLayerNum++)
				{
					Gravisbell::GUID bypassGUID;
					memcpy(&bypassGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
					readBufferByte += sizeof(Gravisbell::GUID);

					layerConnect.lpBypassLayerGUID.push_back(bypassGUID);
				}

				// ���C���[��ڑ�
				this->lpConnectInfo.push_back(layerConnect);
			}
		}

		// �o�̓��C���[GUID
		memcpy(&this->outputLayerGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
		readBufferByte += sizeof(Gravisbell::GUID);

		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
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
		U32 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// ���C���[�f�[�^�̈ꗗ���쐬����
		std::map<Gravisbell::GUID, ILayerData*> lpTmpLayerData;
		{
			// �{�̂�ۗL���Ă��郌�C���[
			for(auto& it : this->lpLayerData)
				lpTmpLayerData[it.first] = it.second;
			// �ڑ����C���[
			for(auto& it : this->lpConnectInfo)
				lpTmpLayerData[it.pLayerData->GetGUID()] = it.pLayerData;
		}


		// ���̓f�[�^�\��
		bufferSize += sizeof(this->inputDataStruct);

		// �ݒ���
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// ���C���[�̐�
		bufferSize += sizeof(U32);

		// �e���C���[�f�[�^
		for(auto& it : lpTmpLayerData)
		{
			// ���C���[��ʃR�[�h
			bufferSize += sizeof(Gravisbell::GUID);

			// ���C���[GUID
			bufferSize += sizeof(Gravisbell::GUID);

			// ���C���[�{��
			bufferSize += it.second->GetUseBufferByteCount();
		}

		// ���C���[�ڑ����
		{
			// ���C���[�ڑ����
			bufferSize += sizeof(U32);

			// ���C���[�ڑ����
			for(auto& it : this->lpConnectInfo)
			{
				// ���C���[��GUID
				bufferSize += sizeof(Gravisbell::GUID);

				// ���C���[�f�[�^��GUID
				bufferSize += sizeof(Gravisbell::GUID);

				// ���̓��C���[�̐�
				bufferSize += sizeof(U32);

				// ���̓��C���[
				bufferSize += sizeof(Gravisbell::GUID) * (U32)it.lpInputLayerGUID.size();

				// �o�C�p�X���̓��C���[�̐�
				bufferSize += sizeof(U32);

				// �o�C�p�X���̓��C���[
				bufferSize += sizeof(Gravisbell::GUID) * (U32)it.lpBypassLayerGUID.size();
			}
		}

		// �o�̓��C���[GUID
		bufferSize += sizeof(Gravisbell::GUID);


		return bufferSize;
	}

	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 FeedforwardNeuralNetwork_LayerData_Base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		// ���C���[�f�[�^�̈ꗗ���쐬����
		std::map<Gravisbell::GUID, ILayerData*> lpTmpLayerData;
		{
			// �{�̂�ۗL���Ă��郌�C���[
			for(auto& it : this->lpLayerData)
				lpTmpLayerData[it.first] = it.second;
			// �ڑ����C���[
			for(auto& it : this->lpConnectInfo)
				lpTmpLayerData[it.pLayerData->GetGUID()] = it.pLayerData;
		}


		int writeBufferByte = 0;

		U32 tmpCount = 0;
		Gravisbell::GUID tmpGUID;

		// ���̓f�[�^�\��
		memcpy(&o_lpBuffer[writeBufferByte], &this->inputDataStruct, sizeof(this->inputDataStruct));
		writeBufferByte += sizeof(this->inputDataStruct);

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// ���C���[�̐�
		tmpCount = (U32)lpTmpLayerData.size();
		memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
		writeBufferByte += sizeof(U32);

		// �e���C���[�f�[�^
		for(auto& it : lpTmpLayerData)
		{
			// ���C���[��ʃR�[�h
			tmpGUID = it.second->GetLayerCode();
			memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
			writeBufferByte += sizeof(Gravisbell::GUID);

			// ���C���[GUID
			tmpGUID = it.second->GetGUID();
			memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
			writeBufferByte += sizeof(Gravisbell::GUID);

			// �e�X�g�p
#ifdef _DEBUG
			U32 useBufferByte  = it.second->GetUseBufferByteCount();
			U32 useBufferByte2 = it.second->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
			if(useBufferByte != useBufferByte2)
			{
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
			}
#endif

			// ���C���[�{��
			writeBufferByte += it.second->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
		}

		// ���C���[�ڑ����
		{
			// ���C���[�ڑ����
			tmpCount = (U32)this->lpConnectInfo.size();
			memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
			writeBufferByte += sizeof(U32);

			// ���C���[�ڑ����
			for(auto& it : this->lpConnectInfo)
			{
				// ���C���[��GUID
				tmpGUID = it.guid;
				memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
				writeBufferByte += sizeof(Gravisbell::GUID);

				// ���C���[�f�[�^��GUID
				tmpGUID = it.pLayerData->GetGUID();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
				writeBufferByte += sizeof(Gravisbell::GUID);

				// ���̓��C���[�̐�
				tmpCount = (U32)it.lpInputLayerGUID.size();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
				writeBufferByte += sizeof(U32);

				// ���̓��C���[
				for(auto guid : it.lpInputLayerGUID)
				{
					memcpy(&o_lpBuffer[writeBufferByte], &guid, sizeof(Gravisbell::GUID));
					writeBufferByte += sizeof(Gravisbell::GUID);
				}

				// �o�C�p�X���̓��C���[�̐�
				tmpCount = (U32)it.lpBypassLayerGUID.size();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
				writeBufferByte += sizeof(U32);

				// �o�C�p�X���̓��C���[
				for(auto guid : it.lpBypassLayerGUID)
				{
					memcpy(&o_lpBuffer[writeBufferByte], &guid, sizeof(Gravisbell::GUID));
					writeBufferByte += sizeof(Gravisbell::GUID);
				}
			}
		}

		// �o�̓��C���[GUID
		tmpGUID = this->outputLayerGUID;
		memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
		writeBufferByte += sizeof(Gravisbell::GUID);


		return writeBufferByte;
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
		auto pConnectLayer = this->GetLayerByGUID(this->outputLayerGUID);
		if(pConnectLayer == NULL)
			return IODataStruct();

		if(const ISingleOutputLayerData* pLayerData = dynamic_cast<const ISingleOutputLayerData*>(pConnectLayer->pLayerData))
		{
			return pLayerData->GetOutputDataStruct();
		}

		return IODataStruct();
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
		ErrorCode err;

		// �S���C���[��ǉ�����
		for(auto it : this->lpConnectInfo)
		{
			err = neuralNetwork.AddLayer(it.pLayerData->CreateLayer(it.guid));
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}

		// ���C���[�Ԃ̐ڑ���ݒ肷��
		for(auto it_connectInfo : this->lpConnectInfo)
		{
			// ���̓��C���[
			for(auto inputGUID : it_connectInfo.lpInputLayerGUID)
			{
				err = neuralNetwork.AddInputLayerToLayer(it_connectInfo.guid, inputGUID);
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;
			}

			// �o�C�p�X���C���[
			for(auto bypassGUID : it_connectInfo.lpBypassLayerGUID)
			{
				err = neuralNetwork.AddBypassLayerToLayer(it_connectInfo.guid, bypassGUID);
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;
			}
		}

		// �o�̓��C���[��ڑ�����
		err = neuralNetwork.SetOutputLayerGUID(this->outputLayerGUID);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

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
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddLayer(const Gravisbell::GUID& i_guid, ILayerData* i_pLayerData)
	{
		// ���C���[������
		if(this->GetLayerByGUID(i_guid))
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// �ǉ�
		this->lpConnectInfo.push_back(LayerConnect(i_guid, i_pLayerData));

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[�f�[�^���폜����.
		@param i_guid	�폜���郌�C���[��GUID */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::EraseLayer(const Gravisbell::GUID& i_guid)
	{
		// �폜���C���[������
		auto it = this->lpConnectInfo.begin();
		while(it != this->lpConnectInfo.end())
		{
			if(it->guid == i_guid)
				break;
		}
		if(it == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		// �폜�Ώۂ̃��C���[����͂Ɏ��ꍇ�͍폜
		for(auto& it_search : this->lpConnectInfo)
		{
			this->EraseInputLayerFromLayer(it_search.guid, i_guid);
			this->EraseBypassLayerFromLayer(it_search.guid, i_guid);
		}

		// ���C���[�f�[�^�{�̂����ꍇ�͍폜
		{
			auto it_pLayerData = this->lpLayerData.find(it->pLayerData->GetGUID());
			if(it_pLayerData != this->lpLayerData.end())
			{
				// �폜�Ώۂ̃��C���[�ȊO�ɓ���̃��C���[�f�[�^���g�p���Ă��郌�C���[�����݂��Ȃ����m�F
				bool onCanErase = true;
				for(auto& it_search : this->lpConnectInfo)
				{
					if(it_search.guid != i_guid)
					{
						if(it_search.pLayerData->GetGUID() == it_pLayerData->first)
						{
							onCanErase = false;
							break;
						}
					}
				}

				// ���C���[�f�[�^���폜
				if(onCanErase)
				{
					delete it_pLayerData->second;
					this->lpLayerData.erase(it_pLayerData);
				}
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
		for(auto it : this->lpLayerData)
		{
			delete it.second;
		}
		this->lpLayerData.clear();

		// �o�͑Ώۃ��C���[�̏ꍇ��������
		this->outputLayerGUID = Gravisbell::GUID();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�^����Ă��郌�C���[�����擾���� */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetLayerCount()
	{
		return (U32)this->lpConnectInfo.size();
	}
	/** ���C���[��GUID��ԍ��w��Ŏ擾���� */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid)
	{
		if(i_layerNum >= this->lpConnectInfo.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		auto it = this->lpConnectInfo.begin();
		for(U32 i=0; i<i_layerNum; i++)
			it++;

		o_guid = it->guid;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�^����Ă��郌�C���[��ԍ��w��Ŏ擾���� */
	FeedforwardNeuralNetwork_LayerData_Base::LayerConnect* FeedforwardNeuralNetwork_LayerData_Base::GetLayerByNum(U32 i_layerNum)
	{
		if(i_layerNum >= this->lpConnectInfo.size())
			return NULL;

		return &this->lpConnectInfo[i_layerNum];
	}
	/** �o�^����Ă��郌�C���[��GUID�w��Ŏ擾���� */
	FeedforwardNeuralNetwork_LayerData_Base::LayerConnect* FeedforwardNeuralNetwork_LayerData_Base::GetLayerByGUID(const Gravisbell::GUID& i_guid)
	{
		for(U32 layerNum=0; layerNum<this->lpConnectInfo.size(); layerNum++)
		{
			if(this->lpConnectInfo[layerNum].guid == i_guid)
				return &this->lpConnectInfo[layerNum];
		}
		return NULL;
	}
	FeedforwardNeuralNetwork_LayerData_Base::LayerConnect* FeedforwardNeuralNetwork_LayerData_Base::GetLayerByGUID(const Gravisbell::GUID& i_guid)const
	{
		return (const_cast<FeedforwardNeuralNetwork_LayerData_Base*>(this))->GetLayerByGUID(i_guid);
	}

	/** �o�^����Ă��郌�C���[�f�[�^��ԍ��w��Ŏ擾���� */
	ILayerData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerDataByNum(U32 i_layerNum)
	{
		Gravisbell::GUID layerGUID;
		ErrorCode err = this->GetLayerGUIDbyNum(i_layerNum, layerGUID);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return NULL;

		return this->GetLayerDataByGUID(layerGUID);
	}
	/** �o�^����Ă��郌�C���[�f�[�^��GUID�w��Ŏ擾���� */
	ILayerData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerDataByGUID(const Gravisbell::GUID& i_guid)
	{
		auto pLayerConnet = this->GetLayerByGUID(i_guid);
		if(pLayerConnet == NULL)
			return NULL;

		return pLayerConnet->pLayerData;
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
		auto pLayerConnet = this->GetLayerByGUID(i_guid);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		if(dynamic_cast<ISingleOutputLayerData*>(pLayerConnet->pLayerData))
		{
			this->outputLayerGUID = i_guid;

			return ErrorCode::ERROR_CODE_NONE;
		}

		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** �o�͐M�����C���[��GUID���擾���� */
	Gravisbell::GUID FeedforwardNeuralNetwork_LayerData_Base::GetOutputLayerGUID()
	{
		return this->outputLayerGUID;
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
		auto pLayerConnet = this->GetLayerByGUID(receiveLayer);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// ���ꃌ�C���[���ǉ��ςłȂ����Ƃ��m�F
		{
			auto it = pLayerConnet->lpInputLayerGUID.begin();
			while(it != pLayerConnet->lpInputLayerGUID.end())
			{
				if(*it == postLayer)
					return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;
				it++;
			}
		}

		// ���C���[��ǉ�
		pLayerConnet->lpInputLayerGUID.push_back(postLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.
		@param	receiveLayer	���͂��󂯎�郌�C���[
		@param	postLayer		���͂�n��(�o�͂���)���C���[. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		// ���C���[�̑��݂��m�F
		auto pLayerConnet = this->GetLayerByGUID(receiveLayer);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// ���ꃌ�C���[���ǉ��ςłȂ����Ƃ��m�F
		{
			auto it = pLayerConnet->lpInputLayerGUID.begin();
			while(it != pLayerConnet->lpInputLayerGUID.end())
			{
				if(*it == postLayer)
					return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;
				it++;
			}
		}

		// ���C���[��ǉ�
		pLayerConnet->lpBypassLayerGUID.push_back(postLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[������̓��C���[���폜����. 
		@param	receiveLayer	���͂��󂯎�郌�C���[
		@param	postLayer		���͂�n��(�o�͂���)���C���[. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::EraseInputLayerFromLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		auto pLayerConnect = this->GetLayerByGUID(receiveLayer);
		if(pLayerConnect == NULL)
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		auto it = pLayerConnect->lpInputLayerGUID.begin();
		while(it != pLayerConnect->lpInputLayerGUID.end())
		{
			if(*it == postLayer)
			{
				it = pLayerConnect->lpInputLayerGUID.erase(it);
			}
			else
			{
				it++;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[����o�C�p�X���C���[���폜����.
		@param	receiveLayer	���͂��󂯎�郌�C���[
		@param	postLayer		���͂�n��(�o�͂���)���C���[. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::EraseBypassLayerFromLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		auto pLayerConnect = this->GetLayerByGUID(receiveLayer);
		if(pLayerConnect == NULL)
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		auto it = pLayerConnect->lpBypassLayerGUID.begin();
		while(it != pLayerConnect->lpBypassLayerGUID.end())
		{
			if(*it == postLayer)
			{
				it = pLayerConnect->lpBypassLayerGUID.erase(it);
			}
			else
			{
				it++;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::ResetInputLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// �폜����
		pLayerConnet->lpInputLayerGUID.clear();

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::ResetBypassLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// �폜����
		pLayerConnet->lpBypassLayerGUID.clear();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return 0;

		return (U32)pLayerConnet->lpInputLayerGUID.size();
	}
	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return 0;

		return (U32)pLayerConnet->lpBypassLayerGUID.size();
	}
	/** ���C���[�ɐڑ����Ă���o�̓��C���[�̐����擾���� */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetOutputLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// ���C���[�̑��݂��m�F
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// �Ώۃ��C���[����̓��C���[�Ɏ����C���[���𐔂���
		U32 outputCount = 0;
		for(auto it : this->lpConnectInfo)
		{
			for(U32 inputNum=0; inputNum<pLayerConnet->lpInputLayerGUID.size(); inputNum++)
			{
				if(pLayerConnet->lpInputLayerGUID[inputNum] == i_layerGUID)
					outputCount++;
			}
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
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// ���̓��C���[�̐����m�F
		if(i_inputNum >= pLayerConnet->lpInputLayerGUID.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// �C�e���[�^�i�s
		auto it = pLayerConnet->lpInputLayerGUID.begin();
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
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// ���̓��C���[�̐����m�F
		if(i_inputNum >= pLayerConnet->lpBypassLayerGUID.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// �C�e���[�^�i�s
		auto it = pLayerConnet->lpBypassLayerGUID.begin();
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
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// �Ώۃ��C���[����̓��C���[�Ɏ����C���[���𐔂��Ĕԍ�����v������I��
		U32 outputNum = 0;
		for(auto it : this->lpConnectInfo)
		{
			for(U32 inputNum=0; inputNum<pLayerConnet->lpInputLayerGUID.size(); inputNum++)
			{
				if(pLayerConnet->lpInputLayerGUID[inputNum] == i_layerGUID)
				{
					if(outputNum == i_outputNum)
					{
						o_postLayerGUID = pLayerConnet->guid;
						return ErrorCode::ERROR_CODE_NONE;
					}
					outputNum++;
				}
			}
		}

		return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell
