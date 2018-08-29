//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[�̃f�[�^
// �����̃��C���[�����A��������
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_LayerData_Base.h"
#include"FeedforwardNeuralNetwork_FUNC.hpp"
#include"FeedforwardNeuralNetwork_Base.h"

#include<boost/uuid/uuid_generators.hpp>

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
			connectInfo.second.pLayerData->Initialize();
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::Initialize(const SettingData::Standard::IData& i_data)
	{
		ErrorCode err;

		// �ݒ���̓o�^
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// �����ۗL�̃��C���[�f�[�^���폜
		for(auto it : this->lpLayerData)
			delete it.second;
		this->lpLayerData.clear();

		// ���̓��C���[�ꗗ���쐬
		this->lpInputLayerGUID.resize(this->layerStructure.inputLayerCount);
		for(size_t i=0; i<this->lpInputLayerGUID.size(); i++)
		{
			this->lpInputLayerGUID[i] = Gravisbell::GUID(boost::uuids::random_generator()().data);
		}

		return this->Initialize();
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize, S64& o_useBufferSize)
	{
		S64 readBufferByte = 0;

		// �ݒ���
		S64 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;

		// ������
		this->Initialize(*pLayerStructure);

		// �ǂݍ��񂾃o�b�t�@���J��
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// ���̓��C���[GUID
		for(size_t i=0; i<this->lpInputLayerGUID.size(); i++)
		{
			Gravisbell::GUID guid;
			memcpy(&guid, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
			readBufferByte += sizeof(Gravisbell::GUID);

			this->lpInputLayerGUID[i] = guid;
		}

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
			S64 useBufferSize = 0;
			auto pLayerData = pLayerDLL->CreateLayerDataFromBuffer(guid, &i_lpBuffer[readBufferByte], i_bufferSize - readBufferByte, useBufferSize);
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

				// ���C���[�̌Œ�t���O
				bool onFixFlag = false;
				memcpy(&onFixFlag, &i_lpBuffer[readBufferByte], sizeof(onFixFlag));
				readBufferByte += sizeof(onFixFlag);

				// ���C���[�f�[�^��GUID
				Gravisbell::GUID layerDataGUID;
				memcpy(&layerDataGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
				readBufferByte += sizeof(Gravisbell::GUID);

				// ���C���[�f�[�^�擾
				auto pLayerData = this->lpLayerData[layerDataGUID];
				if(pLayerData == NULL)
					return ErrorCode::ERROR_CODE_LAYER_CREATE;

				// ���C���[�ڑ������쐬
				LayerConnect layerConnect(layerGUID, pLayerData, onFixFlag);

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
				this->lpConnectInfo[layerConnect.guid] = layerConnect;
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
	U64 FeedforwardNeuralNetwork_LayerData_Base::GetUseBufferByteCount()const
	{
		U64 bufferSize = 0;

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
				lpTmpLayerData[it.second.pLayerData->GetGUID()] = it.second.pLayerData;
		}


		// �ݒ���
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// ���̓��C���[GUID�ꗗ
		bufferSize += this->layerStructure.inputLayerCount * sizeof(Gravisbell::GUID);

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

				// ���C���[�̌Œ�t���O
				bufferSize += sizeof(bool);

				// ���C���[�f�[�^��GUID
				bufferSize += sizeof(Gravisbell::GUID);

				// ���̓��C���[�̐�
				bufferSize += sizeof(U32);

				// ���̓��C���[
				bufferSize += sizeof(Gravisbell::GUID) * (U32)it.second.lpInputLayerGUID.size();

				// �o�C�p�X���̓��C���[�̐�
				bufferSize += sizeof(U32);

				// �o�C�p�X���̓��C���[
				bufferSize += sizeof(Gravisbell::GUID) * (U32)it.second.lpBypassLayerGUID.size();
			}
		}

		// �o�̓��C���[GUID
		bufferSize += sizeof(Gravisbell::GUID);


		return bufferSize;
	}

	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S64 FeedforwardNeuralNetwork_LayerData_Base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return -1;

		// ���C���[�f�[�^�̈ꗗ���쐬����
		std::map<Gravisbell::GUID, ILayerData*> lpTmpLayerData;
		{
			// �{�̂�ۗL���Ă��郌�C���[
			for(auto& it : this->lpLayerData)
				lpTmpLayerData[it.first] = it.second;
			// �ڑ����C���[
			for(auto& it : this->lpConnectInfo)
				lpTmpLayerData[it.second.pLayerData->GetGUID()] = it.second.pLayerData;
		}


		S64 writeBufferByte = 0;

		U32 tmpCount = 0;
		Gravisbell::GUID tmpGUID;

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// ���̓��C���[GUID
		for(auto guid : this->lpInputLayerGUID)
		{
			memcpy(&o_lpBuffer[writeBufferByte], &guid, sizeof(Gravisbell::GUID));
			writeBufferByte += sizeof(Gravisbell::GUID);
		}

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
			U64 useBufferByte  = it.second->GetUseBufferByteCount();
			S64 useBufferByte2 = it.second->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
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
				tmpGUID = it.second.guid;
				memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
				writeBufferByte += sizeof(Gravisbell::GUID);

				// ���C���[�̌Œ�t���O
				memcpy(&o_lpBuffer[writeBufferByte], &it.second.onFixFlag, sizeof(it.second.onFixFlag));
				writeBufferByte += sizeof(it.second.onFixFlag);

				// ���C���[�f�[�^��GUID
				tmpGUID = it.second.pLayerData->GetGUID();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
				writeBufferByte += sizeof(Gravisbell::GUID);

				// ���̓��C���[�̐�
				tmpCount = (U32)it.second.lpInputLayerGUID.size();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
				writeBufferByte += sizeof(U32);

				// ���̓��C���[
				for(auto guid : it.second.lpInputLayerGUID)
				{
					memcpy(&o_lpBuffer[writeBufferByte], &guid, sizeof(Gravisbell::GUID));
					writeBufferByte += sizeof(Gravisbell::GUID);
				}

				// �o�C�p�X���̓��C���[�̐�
				tmpCount = (U32)it.second.lpBypassLayerGUID.size();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
				writeBufferByte += sizeof(U32);

				// �o�C�p�X���̓��C���[
				for(auto guid : it.second.lpBypassLayerGUID)
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

		// �\���̂ɓǂݍ���
		this->pLayerStructure->WriteToStruct((BYTE*)&this->layerStructure);

		return ERROR_CODE_NONE;
	}
	/** ���C���[�̐ݒ�����擾���� */
	const SettingData::Standard::IData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}

	//===========================
	// ���C���[�\��
	//===========================
	/** ���̓f�[�^�\�����g�p�\���m�F����.
		@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
		@return	�g�p�\�ȓ��̓f�[�^�\���̏ꍇtrue���Ԃ�. */
	bool FeedforwardNeuralNetwork_LayerData_Base::CheckCanUseInputDataStruct(const Gravisbell::GUID& i_guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		return this->GetOutputDataStruct(i_guid, i_lpInputDataStruct, i_inputLayerCount).GetDataCount() != 0;
	}
	bool FeedforwardNeuralNetwork_LayerData_Base::CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		return this->GetOutputDataStruct(i_lpInputDataStruct, i_inputLayerCount).GetDataCount() != 0;
	}


	/** �o�̓f�[�^�\�����擾����.
		@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
		@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
	IODataStruct FeedforwardNeuralNetwork_LayerData_Base::GetOutputDataStruct(const Gravisbell::GUID& i_guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		// ���̓��C���[�����قȂ�ꍇ�͏I��
		if(i_inputLayerCount != this->layerStructure.inputLayerCount)
			return IODataStruct(0,0,0,0);

		this->tmp_lpInputDataStruct = i_lpInputDataStruct;
		this->tmp_inputLayerCount = i_inputLayerCount;
		this->tmp_lpOutputDataStruct.clear();

		return this->GetOutputDataStruct(i_guid);
	}
	/** �o�̓f�[�^�\�����擾����.
		@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
		@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
	IODataStruct FeedforwardNeuralNetwork_LayerData_Base::GetOutputDataStruct(const Gravisbell::GUID& i_guid)
	{
		// ���Ɍv�Z�ς݂̃��C���[���m�F����
		{
			auto it_layer = this->tmp_lpOutputDataStruct.find(i_guid);
			if(it_layer != this->tmp_lpOutputDataStruct.end())
				return it_layer->second;
		}

		for(size_t inputNum=0; inputNum<this->lpInputLayerGUID.size(); inputNum++)
		{
			if(i_guid == this->lpInputLayerGUID[inputNum])
			{
				return this->tmp_lpOutputDataStruct[i_guid] = this->tmp_lpInputDataStruct[inputNum];
			}
		}

		auto pConnectLayer = this->GetLayerByGUID(i_guid);
		if(pConnectLayer == NULL)
			return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);

		// ���̓��C���[
		std::vector<IODataStruct> lpInputDataStruct(pConnectLayer->lpInputLayerGUID.size());
		for(U32 inputLayerNum=0; inputLayerNum<lpInputDataStruct.size(); inputLayerNum++)
		{
			lpInputDataStruct[inputLayerNum] = this->GetOutputDataStruct(pConnectLayer->lpInputLayerGUID[inputLayerNum]);
			if(lpInputDataStruct[inputLayerNum].GetDataCount() == 0)
				return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);
		}
		// �o�C�p�X���C���[
		std::vector<IODataStruct> lpBypassDataStruct(pConnectLayer->lpBypassLayerGUID.size());
		for(U32 inputLayerNum=0; inputLayerNum<lpBypassDataStruct.size(); inputLayerNum++)
		{
			lpBypassDataStruct[inputLayerNum] = this->GetOutputDataStruct(pConnectLayer->lpBypassLayerGUID[inputLayerNum]);
			if(lpBypassDataStruct[inputLayerNum].GetDataCount() == 0)
				return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);
		}

		if(lpInputDataStruct.size() == 0)
			return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);

		// ���͂���������ꍇ
		if(lpInputDataStruct.size() > 1)
		{
			// �������͂��󂯕t���Ă��邩�`�F�b�N
			if(pConnectLayer->pLayerData->CheckCanUseInputDataStruct(&lpInputDataStruct[0], (U32)lpInputDataStruct.size()))
			{
				return this->tmp_lpOutputDataStruct[i_guid] = pConnectLayer->pLayerData->GetOutputDataStruct(&lpInputDataStruct[0], (U32)lpInputDataStruct.size());
			}
			else
			{
				// CH�ȊO����v���Ă��邱�Ƃ��m�F
				IODataStruct inputDataStruct = lpInputDataStruct[0];
				for(U32 layerNum=1; layerNum<lpInputDataStruct.size(); layerNum++)
				{
					if(inputDataStruct.x != lpInputDataStruct[layerNum].x)	return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);
					if(inputDataStruct.y != lpInputDataStruct[layerNum].y)	return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);
					if(inputDataStruct.z != lpInputDataStruct[layerNum].z)	return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);

					inputDataStruct.ch += lpInputDataStruct[layerNum].ch;
				}

				return this->tmp_lpOutputDataStruct[i_guid] = pConnectLayer->pLayerData->GetOutputDataStruct(&inputDataStruct, 1);
			}
		}

		return this->tmp_lpOutputDataStruct[i_guid] = pConnectLayer->pLayerData->GetOutputDataStruct(&lpInputDataStruct[0], 1);
	}

	/** �o�̓f�[�^�\�����擾����.
		@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
		@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
	IODataStruct FeedforwardNeuralNetwork_LayerData_Base::GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(i_inputLayerCount == 0)
			return false;
		if(i_inputLayerCount > 1)
			return false;
		if(i_lpInputDataStruct == NULL)
			return false;

		return this->GetOutputDataStruct(this->outputLayerGUID, i_lpInputDataStruct, i_inputLayerCount);
	}

	/** �����o�͂��\�����m�F���� */
	bool FeedforwardNeuralNetwork_LayerData_Base::CheckCanHaveMultOutputLayer(void)
	{
		return false;
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** �쐬���ꂽ�V�K�j���[�����l�b�g���[�N�ɑ΂��ē������C���[��ǉ����� */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddConnectionLayersToNeuralNetwork(class FeedforwardNeuralNetwork_Base& neuralNetwork, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		ErrorCode err;

		// �o�͕������C���[�̎���ID
		static const Gravisbell::GUID SEPARATE_LAYER_GUID(0xc13c30da, 0x056e, 0x46d0, 0x90, 0xfc, 0x60, 0x87, 0x66, 0xfb, 0x43, 0x2e);

		std::map<Gravisbell::GUID, Gravisbell::GUID>	lpSubstitutionLayer;	// ��փ��C���[<�����C���[GUID, ��փ��C���[GUID>

		// �S���C���[��ǉ�����
		for(auto it : this->lpConnectInfo)
		{
			// �Ώۃ��C���[�ɑ΂�����̓f�[�^�\���ꗗ���쐬
			std::vector<IODataStruct> lpInputDataStruct;
			for(auto inputGUID : it.second.lpInputLayerGUID)
			{
				lpInputDataStruct.push_back(this->GetOutputDataStruct(inputGUID, i_lpInputDataStruct, i_inputLayerCount));
			}
			
			err = neuralNetwork.AddLayer(it.second.pLayerData->CreateLayer(it.second.guid, &lpInputDataStruct[0], (U32)lpInputDataStruct.size(), neuralNetwork.GetTemporaryMemoryManager()), it.second.onFixFlag);

			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}

		// �����o�͂������C���[�ŁA�P��o�͂̔\�͂��������Ȃ����̂�T���A��փ��C���[���쐬����
		for(auto it : this->lpConnectInfo)
		{
			if(this->GetOutputLayerCount(it.second.guid) > 1)
			{
				ILayerData* pLayerData = it.second.pLayerData;
				if(!pLayerData->CheckCanHaveMultOutputLayer())
				{
					// �P��o�͋@�\���������Ȃ��̂ɕ����o�͂������Ă���

					// �o�͕������C���|��DLL���擾
					auto pDLL = this->layerDLLManager.GetLayerDLLByGUID(SEPARATE_LAYER_GUID);
					if(pDLL == NULL)
						return ErrorCode::ERROR_CODE_DLL_NOTFOUND;

					// ���C���[�\�������쐬
					auto pLayerStructure = pDLL->CreateLayerStructureSetting();
					if(pLayerStructure == NULL)
						return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
					auto pItem = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Int*>(pLayerStructure->GetItemByID(L"separateCount"));
					if(pItem == NULL)
						return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;

					pItem->SetValue(this->GetOutputLayerCount(it.second.guid));

					// ���C���[�f�[�^���쐬
					auto pSubstitutionLayerData = pDLL->CreateLayerData(*pLayerStructure);
					delete pLayerStructure;

					// ���C���[��ǉ�
					ILayerBase* pSubstitutionLayer = NULL;
					err = neuralNetwork.AddTemporaryLayer(pSubstitutionLayerData, &pSubstitutionLayer, &this->GetOutputDataStruct(it.second.guid, i_lpInputDataStruct, i_inputLayerCount), 1, true);
					if(err != ErrorCode::ERROR_CODE_NONE)
						return err;

					lpSubstitutionLayer[it.second.guid] = pSubstitutionLayer->GetGUID();

					// ��փ��C���[�̓��͂��֐惌�C���[�ɐݒ�
					err = neuralNetwork.AddInputLayerToLayer(pSubstitutionLayer->GetGUID(), it.second.guid);
					if(err != ErrorCode::ERROR_CODE_NONE)
						return err;
				}
			}
		}
		// ���̓��C���[�𕡐����C���[����g�p���Ă��Ȃ����m�F����
		for(S32 inputLyaerNum=0; inputLyaerNum<this->layerStructure.inputLayerCount; inputLyaerNum++)
		{
			if(this->GetOutputLayerCount(this->GetInputGUID(inputLyaerNum)) > 1)
			{
				// �o�͕������C���|��DLL���擾
				auto pDLL = this->layerDLLManager.GetLayerDLLByGUID(SEPARATE_LAYER_GUID);
				if(pDLL == NULL)
					return ErrorCode::ERROR_CODE_DLL_NOTFOUND;

				// ���C���[�\�������쐬
				auto pLayerStructure = pDLL->CreateLayerStructureSetting();
				if(pLayerStructure == NULL)
					return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
				auto pItem = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Int*>(pLayerStructure->GetItemByID(L"separateCount"));
				if(pItem == NULL)
					return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;

				pItem->SetValue(this->GetOutputLayerCount(this->GetInputGUID(inputLyaerNum)));

				// ���C���[�f�[�^���쐬
				auto pSubstitutionLayerData = pDLL->CreateLayerData(*pLayerStructure);
				delete pLayerStructure;

				// ���C���[��ǉ�
				ILayerBase* pSubstitutionLayer = NULL;
				err = neuralNetwork.AddTemporaryLayer(pSubstitutionLayerData, &pSubstitutionLayer, i_lpInputDataStruct, i_inputLayerCount, true);
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				lpSubstitutionLayer[this->GetInputGUID(inputLyaerNum)] = pSubstitutionLayer->GetGUID();

				// ��փ��C���[�̓��͂��֐惌�C���[�ɐݒ�
				err = neuralNetwork.AddInputLayerToLayer(pSubstitutionLayer->GetGUID(), this->GetInputGUID(inputLyaerNum));
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;
			}
		}


		// ���C���[�Ԃ̐ڑ���ݒ肷��
		for(auto it_connectInfo : this->lpConnectInfo)
		{
			// ���̓��C���[
			for(auto inputGUID : it_connectInfo.second.lpInputLayerGUID)
			{
				if(lpSubstitutionLayer.count(inputGUID))
				{
					// ��փ��C���[���g�p����
					err = neuralNetwork.AddInputLayerToLayer(it_connectInfo.second.guid, lpSubstitutionLayer[inputGUID]);
					if(err != ErrorCode::ERROR_CODE_NONE)
						return err;
				}
				else
				{
					err = neuralNetwork.AddInputLayerToLayer(it_connectInfo.second.guid, inputGUID);
					if(err != ErrorCode::ERROR_CODE_NONE)
						return err;
				}
			}

			// �o�C�p�X���C���[
			for(auto bypassGUID : it_connectInfo.second.lpBypassLayerGUID)
			{
				err = neuralNetwork.AddBypassLayerToLayer(it_connectInfo.second.guid, bypassGUID);
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
		@param	i_pLayerData	�ǉ����郌�C���[�f�[�^�̃A�h���X.
		@param	i_onFixFlag		���C���[���Œ艻����t���O. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddLayer(const Gravisbell::GUID& i_guid, ILayerData* i_pLayerData, bool i_onFixFlag)
	{
		// ���C���[������
		if(this->GetLayerByGUID(i_guid))
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// �ǉ�
		this->lpConnectInfo[i_guid] = LayerConnect(i_guid, i_pLayerData, i_onFixFlag);

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
			if(it->second.guid == i_guid)
				break;
		}
		if(it == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		// �폜�Ώۂ̃��C���[����͂Ɏ��ꍇ�͍폜
		for(auto& it_search : this->lpConnectInfo)
		{
			this->EraseInputLayerFromLayer(it_search.second.guid, i_guid);
			this->EraseBypassLayerFromLayer(it_search.second.guid, i_guid);
		}

		// ���C���[�f�[�^�{�̂����ꍇ�͍폜
		{
			auto it_pLayerData = this->lpLayerData.find(it->second.pLayerData->GetGUID());
			if(it_pLayerData != this->lpLayerData.end())
			{
				// �폜�Ώۂ̃��C���[�ȊO�ɓ���̃��C���[�f�[�^���g�p���Ă��郌�C���[�����݂��Ȃ����m�F
				bool onCanErase = true;
				for(auto& it_search : this->lpConnectInfo)
				{
					if(it_search.second.guid != i_guid)
					{
						if(it_search.second.pLayerData->GetGUID() == it_pLayerData->first)
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

		o_guid = it->second.guid;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�^����Ă��郌�C���[��ԍ��w��Ŏ擾���� */
	FeedforwardNeuralNetwork_LayerData_Base::LayerConnect* FeedforwardNeuralNetwork_LayerData_Base::GetLayerByNum(U32 i_layerNum)
	{
		if(i_layerNum >= this->lpConnectInfo.size())
			return NULL;

		auto it = this->lpConnectInfo.begin();
		for(U32 i=0; i<i_layerNum; i++)
			it++;

		return &it->second;
	}
	/** �o�^����Ă��郌�C���[��GUID�w��Ŏ擾���� */
	FeedforwardNeuralNetwork_LayerData_Base::LayerConnect* FeedforwardNeuralNetwork_LayerData_Base::GetLayerByGUID(const Gravisbell::GUID& i_guid)
	{
		auto it = this->lpConnectInfo.find(i_guid);
		if(it == this->lpConnectInfo.end())
			return NULL;

		return &it->second;
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

	/** ���C���[�̌Œ艻�t���O���擾���� */
	bool FeedforwardNeuralNetwork_LayerData_Base::GetLayerFixFlagByGUID(const Gravisbell::GUID& i_guid)
	{
		auto pLayerConnet = this->GetLayerByGUID(i_guid);
		if(pLayerConnet == NULL)
			return false;

		return pLayerConnet->onFixFlag;
	}


	//====================================
	// ���o�̓��C���[
	//====================================
	/** ���͐M�����C���[�����擾���� */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetInputCount()
	{
		return this->layerStructure.inputLayerCount;
	}
	/** ���͐M���Ɋ��蓖�Ă��Ă���GUID���擾���� */
	Gravisbell::GUID FeedforwardNeuralNetwork_LayerData_Base::GetInputGUID(U32 i_inputLayerNum)
	{
		if(i_inputLayerNum >= this->lpInputLayerGUID.size())
			return Gravisbell::GUID();
		return this->lpInputLayerGUID[i_inputLayerNum];
	}

	/** �o�͐M�����C���[��ݒ肷�� */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetOutputLayerGUID(const Gravisbell::GUID& i_guid)
	{
		auto pLayerConnet = this->GetLayerByGUID(i_guid);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		if(pLayerConnet->pLayerData && !pLayerConnet->pLayerData->CheckCanHaveMultOutputLayer())
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
		// �Ώۃ��C���[����̓��C���[�Ɏ����C���[���𐔂���
		U32 outputCount = 0;
		for(auto it : this->lpConnectInfo)
		{
			for(U32 inputNum=0; inputNum<it.second.lpInputLayerGUID.size(); inputNum++)
			{
				if(it.second.lpInputLayerGUID[inputNum] == i_layerGUID)
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


	//===========================
	// �I�v�e�B�}�C�U�[�ݒ�
	//===========================
	/** �I�v�e�B�}�C�U�[��ύX���� */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::ChangeOptimizer(const wchar_t i_optimizerID[])
	{
		for(auto& it : this->lpConnectInfo)
		{
			if(it.second.pLayerData)
				it.second.pLayerData->ChangeOptimizer(i_optimizerID);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
	{
		for(auto& it : this->lpConnectInfo)
		{
			if(it.second.pLayerData)
				it.second.pLayerData->SetOptimizerHyperParameter(i_parameterID, i_value);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
	{
		for(auto& it : this->lpConnectInfo)
		{
			if(it.second.pLayerData)
				it.second.pLayerData->SetOptimizerHyperParameter(i_parameterID, i_value);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
	{
		for(auto& it : lpLayerData)
		{
			it.second->SetOptimizerHyperParameter(i_parameterID, i_value);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
