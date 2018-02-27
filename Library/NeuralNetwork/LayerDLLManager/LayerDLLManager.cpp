// LayerDLLManager.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"
#include "Layer/NeuralNetwork/NNlayerFunction.h"
#include "Library/NeuralNetwork/LayerDLLManager.h"

#include<string>
#include<vector>

#include<boost/uuid/uuid_generators.hpp>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** DLL�N���X */
	class NNLayerDLL : public ILayerDLL
	{
	protected:
		HMODULE hModule;

		FuncGetLayerCode funcGetLayerCode;
		FuncGetVersionCode funcGetVersionCode;
		
		FuncCreateLayerStructureSetting				funcCreateLayerStructureSetting;
		FuncCreateLayerStructureSettingFromBuffer	funcCreateLayerStructureSettingFromBuffer;

		FuncCreateLayerRuntimeParameter				funcCreateRuntimeParameter;
		FuncCreateLayerRuntimeParameterFromBuffer	funcCreateRuntimeParameterFromBuffer;

		FuncCreateLayerData				funcCreateLayerData;
		FuncCreateLayerDataFromBuffer	funcCreateLayerDataFromBuffer;

		const ILayerDLLManager& layerDLLManager;

	public:
		/** �R���X�g���N�^ */
		NNLayerDLL(const ILayerDLLManager& i_layerDLLManager)
			:	hModule	(NULL)
			,	funcGetLayerCode							(NULL)
			,	funcGetVersionCode							(NULL)
			,	funcCreateLayerStructureSetting				(NULL)
			,	funcCreateLayerStructureSettingFromBuffer	(NULL)
			,	funcCreateRuntimeParameter					(NULL)
			,	funcCreateRuntimeParameterFromBuffer		(NULL)
			,	funcCreateLayerData							(NULL)
			,	funcCreateLayerDataFromBuffer				(NULL)
			,	layerDLLManager								(i_layerDLLManager)
		{
		}
		/** �f�X�g���N�^ */
		~NNLayerDLL()
		{
			if(this->hModule != NULL)
			{
				FreeLibrary(this->hModule);
				this->hModule = NULL;
			}
		}

	public:
		/** ���C���[���ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)const
		{
			if(this->funcGetLayerCode == NULL)
				return ERROR_CODE_DLL_LOAD_FUNCTION;

			return this->funcGetLayerCode(o_layerCode);
		}
		/** �o�[�W�����R�[�h���擾����.
			@param o_versionCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		ErrorCode GetVersionCode(VersionCode& o_versionCode)const
		{
			if(this->funcGetVersionCode == NULL)
				return ERROR_CODE_DLL_LOAD_FUNCTION;

			return this->funcGetVersionCode(o_versionCode);
		}

		//==============================
		// ���C���[�\���쐬
		//==============================
	public:
		/** ���C���[�\���ݒ���쐬���� */
		SettingData::Standard::IData* CreateLayerStructureSetting(void)const
		{
			if(this->funcCreateLayerStructureSetting == NULL)
				return NULL;

			return this->funcCreateLayerStructureSetting();
		}
		/** ���C���[�\���ݒ���쐬����
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize)const
		{
			if(this->funcCreateLayerStructureSettingFromBuffer == NULL)
				return NULL;

			return this->funcCreateLayerStructureSettingFromBuffer(i_lpBuffer, i_bufferSize, o_useBufferSize);
		}


		//==============================
		// �w�K�ݒ�쐬
		//==============================
	public:
		/** ���C���[�w�K�ݒ���쐬���� */
		SettingData::Standard::IData* CreateRuntimeParameter(void)const
		{
			if(this->funcCreateRuntimeParameter == NULL)
				return NULL;

			return this->funcCreateRuntimeParameter();
		}
		/** ���C���[�w�K�ݒ���쐬����
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		SettingData::Standard::IData* CreateRuntimeParameterFromBuffer(const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize)const
		{
			if(this->funcCreateRuntimeParameterFromBuffer == NULL)
				return NULL;

			return this->funcCreateRuntimeParameterFromBuffer(i_lpBuffer, i_bufferSize, o_useBufferSize);
		}


		//==============================
		// ���C���[�쐬
		//==============================
	public:
		/** ���C���[�f�[�^���쐬.
			GUID�͎������蓖��.
			@param	i_layerStructure	���C���[�\��.
			@param	i_inputDataStruct	���̓f�[�^�\��. */
		ILayerData* CreateLayerData(const SettingData::Standard::IData& i_layerStructure)const
		{
			boost::uuids::uuid uuid = boost::uuids::random_generator()();

			return this->CreateLayerData(uuid.data, i_layerStructure);
		}
		/** ���C���[�f�[�^���쐬
			@param guid	�쐬���C���[��GUID
			@param	i_layerStructure	���C���[�\��.
			@param	i_inputDataStruct	���̓f�[�^�\��. */
		ILayerData* CreateLayerData(const Gravisbell::GUID& guid, const SettingData::Standard::IData& i_layerStructure)const
		{
			if(this->funcCreateLayerData == NULL)
				return NULL;

			return this->funcCreateLayerData(&this->layerDLLManager, guid, i_layerStructure);
		}


		/** ���C���[�f�[�^���쐬.
			GUID�͎������蓖��.
			@param	i_layerStructure	���C���[�\��.
			@param	i_inputDataStruct	���̓f�[�^�\��. */
		ILayerData* CreateLayerDataFromBuffer(const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize)const
		{
			boost::uuids::uuid uuid = boost::uuids::random_generator()();

			return this->CreateLayerDataFromBuffer(uuid.data, i_lpBuffer, i_bufferSize, o_useBufferSize);
		}
		/** ���C���[�f�[�^���쐬
			@param guid	�쐬���C���[��GUID
			@param	i_layerStructure	���C���[�\��.
			@param	i_inputDataStruct	���̓f�[�^�\��. */
		ILayerData* CreateLayerDataFromBuffer(const Gravisbell::GUID& guid, const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize)const
		{
			if(this->funcCreateLayerDataFromBuffer == NULL)
				return NULL;

			return this->funcCreateLayerDataFromBuffer(&this->layerDLLManager, guid, i_lpBuffer, i_bufferSize, o_useBufferSize);
		}

	private:
		/** DLL���t�@�C������쐬����(���ʕ��������o��) */
		static NNLayerDLL* CreateFromFileCommon(const ::std::wstring& filePath, const ILayerDLLManager& i_layerDLLManager)
		{
			// �o�b�t�@���쐬
			NNLayerDLL* pLayerDLL = new NNLayerDLL(i_layerDLLManager);
			if(pLayerDLL == NULL)
				return NULL;

			do
			{
				// DLL�̓ǂݍ���
				pLayerDLL->hModule = LoadLibrary(filePath.c_str());
				if(pLayerDLL->hModule == NULL)
				{
					DWORD errNo = GetLastError();
					wchar_t szErrorMessage[1024];

					FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM,
								NULL,
								errNo,
								0,
								szErrorMessage,
								sizeof(szErrorMessage)/sizeof(szErrorMessage[0]),
								NULL);

					break;
				}

				// �֐��ǂݍ���
				// ���C���[�R�[�h
				pLayerDLL->funcGetLayerCode = (FuncGetLayerCode)GetProcAddress(pLayerDLL->hModule, "GetLayerCode");
				if(pLayerDLL->funcGetLayerCode == NULL)
					break;
				// �o�[�W�����R�[�h
				pLayerDLL->funcGetVersionCode = (FuncGetVersionCode)GetProcAddress(pLayerDLL->hModule, "GetVersionCode");
				if(pLayerDLL->funcGetVersionCode == NULL)
					break;

				// ���C���[�\��
				pLayerDLL->funcCreateLayerStructureSetting = (FuncCreateLayerStructureSetting)GetProcAddress(pLayerDLL->hModule, "CreateLayerStructureSetting");
				if(pLayerDLL->funcCreateLayerStructureSetting == NULL)
					break;
				pLayerDLL->funcCreateLayerStructureSettingFromBuffer = (FuncCreateLayerStructureSettingFromBuffer)GetProcAddress(pLayerDLL->hModule, "CreateLayerStructureSettingFromBuffer");
				if(pLayerDLL->funcCreateLayerStructureSettingFromBuffer == NULL)
					break;

				// �w�K�ݒ�
				pLayerDLL->funcCreateRuntimeParameter = (FuncCreateLayerStructureSetting)GetProcAddress(pLayerDLL->hModule, "CreateRuntimeParameter");
				if(pLayerDLL->funcCreateRuntimeParameter == NULL)
					break;
				pLayerDLL->funcCreateRuntimeParameterFromBuffer = (FuncCreateLayerStructureSettingFromBuffer)GetProcAddress(pLayerDLL->hModule, "CreateRuntimeParameterFromBuffer");
				if(pLayerDLL->funcCreateRuntimeParameterFromBuffer == NULL)
					break;

				return pLayerDLL;
			}
			while(0);


			// DLL�̍쐬�Ɏ��s.�o�b�t�@���폜
			delete pLayerDLL;

			return NULL;
		}

	public:
		/** DLL���t�@�C������쐬���� */
		static NNLayerDLL* CreateFromFileCPU(const ::std::wstring& filePath, const ILayerDLLManager& i_layerDLLManager)
		{
			// ���ʕ������쐬
			NNLayerDLL* pLayerDLL = CreateFromFileCommon(filePath, i_layerDLLManager);
			if(pLayerDLL == NULL)
				return NULL;

			do
			{
				// ���C���[�f�[�^�쐬
				pLayerDLL->funcCreateLayerData = (FuncCreateLayerData)GetProcAddress(pLayerDLL->hModule, "CreateLayerDataCPU");
				if(pLayerDLL->funcCreateLayerData == NULL)
					break;
				// ���C���[�f�[�^�쐬
				pLayerDLL->funcCreateLayerDataFromBuffer = (FuncCreateLayerDataFromBuffer)GetProcAddress(pLayerDLL->hModule, "CreateLayerDataCPUfromBuffer");
				if(pLayerDLL->funcCreateLayerDataFromBuffer == NULL)
					break;


				return pLayerDLL;
			}
			while(0);


			// DLL�̍쐬�Ɏ��s.�o�b�t�@���폜
			delete pLayerDLL;

			return NULL;
		}
		/** DLL���t�@�C������쐬���� */
		static NNLayerDLL* CreateFromFileGPU(const ::std::wstring& filePath, const ILayerDLLManager& i_layerDLLManager)
		{
			// ���ʕ������쐬
			NNLayerDLL* pLayerDLL = CreateFromFileCommon(filePath, i_layerDLLManager);
			if(pLayerDLL == NULL)
				return NULL;

			do
			{
				// ���C���[�f�[�^�쐬
				pLayerDLL->funcCreateLayerData= (FuncCreateLayerData)GetProcAddress(pLayerDLL->hModule, "CreateLayerDataGPU");
				if(pLayerDLL->funcCreateLayerData == NULL)
					break;
				// ���C���[�f�[�^�쐬
				pLayerDLL->funcCreateLayerDataFromBuffer = (FuncCreateLayerDataFromBuffer)GetProcAddress(pLayerDLL->hModule, "CreateLayerDataGPUfromBuffer");
				if(pLayerDLL->funcCreateLayerDataFromBuffer == NULL)
					break;

				return pLayerDLL;
			}
			while(0);


			// DLL�̍쐬�Ɏ��s.�o�b�t�@���폜
			delete pLayerDLL;

			return NULL;
		}
	};


	/** DLL�Ǘ��N���X */
	class LayerDLLManagerBase : public ILayerDLLManager
	{
	protected:
		std::vector<NNLayerDLL*> lppNNLayerDLL;

	public:
		/** �R���X�g���N�^ */
		LayerDLLManagerBase()
		{
		}
		/** �f�X�g���N�^ */
		virtual ~LayerDLLManagerBase()
		{
			for(auto it : this->lppNNLayerDLL)
			{
				if(it != NULL)
					delete it;
			}
		}

	public:
		/** DLL��ǂݍ���ŁA�Ǘ��ɒǉ�����.
			@param szFilePath	�ǂݍ��ރt�@�C���̃p�X.
			@param o_addLayerCode	�ǉ����ꂽGUID�̊i�[��A�h���X.
			@return	���������ꍇ0���Ԃ�. */
		virtual ErrorCode ReadLayerDLL(const wchar_t szFilePath[], Gravisbell::GUID& o_addLayerCode) = 0;
		/** DLL��ǂݍ���ŁA�Ǘ��ɒǉ�����.
			@param szFilePath	�ǂݍ��ރt�@�C���̃p�X.
			@return	���������ꍇ0���Ԃ�. */
		ErrorCode ReadLayerDLL(const wchar_t szFilePath[])
		{
			Gravisbell::GUID layerCode;
			return this->ReadLayerDLL(szFilePath, layerCode);
		}

		/** �Ǘ����Ă��郌�C���[DLL�̐����擾���� */
		unsigned int GetLayerDLLCount()const
		{
			return (U32)this->lppNNLayerDLL.size();
		}
		/** �Ǘ����Ă��郌�C���[DLL��ԍ��w��Ŏ擾����.
			@param	num	�擾����DLL�̊Ǘ��ԍ�.
			@return ���������ꍇ��DLL�N���X�̃A�h���X. ���s�����ꍇ��NULL */
		const ILayerDLL* GetLayerDLLByNum(unsigned int num)const
		{
			if(num >= this->lppNNLayerDLL.size())
				return NULL;

			return this->lppNNLayerDLL[num];
		}
		/** �Ǘ����Ă��郌�C���[DLL��guid�w��Ŏ擾����.
			@param guid	�擾����DLL��GUID.
			@return ���������ꍇ��DLL�N���X�̃A�h���X. ���s�����ꍇ��NULL */
		const ILayerDLL* GetLayerDLLByGUID(Gravisbell::GUID i_layerCode)const
		{
			for(unsigned int i=0; i<this->lppNNLayerDLL.size(); i++)
			{
				auto pLayerDLL = this->lppNNLayerDLL[i];
				if(pLayerDLL == NULL)
					continue;

				// GUID���擾����
				Gravisbell::GUID layerCode;
				if(pLayerDLL->GetLayerCode(layerCode) != 0)
					continue;

				// �m�F
				if(layerCode == i_layerCode)
					return pLayerDLL;
			}

			return NULL;
		}

		/** ���C���[DLL���폜����. */
		ErrorCode EraseLayerDLL(Gravisbell::GUID i_layerCode)
		{
			auto it = this->lppNNLayerDLL.begin();
			while(it != this->lppNNLayerDLL.end())
			{
				if(*it == NULL)
				{
					it++;
					continue;
				}

				// GUID���擾����
				Gravisbell::GUID layerCode;
				if((*it)->GetLayerCode(layerCode) != 0)
				{
					it++;
					continue;
				}

				// �m�F
				if(layerCode != i_layerCode)
				{
					it++;
					continue;
				}

				// �폜
				delete *it;
				this->lppNNLayerDLL.erase(it);
				return ERROR_CODE_NONE;
			}
			return ERROR_CODE_DLL_ERASE_NOTFOUND;
		}
	};

	/** DLL�Ǘ��N���X(CPU����) */
	class LayerDLLManagerCPU : public LayerDLLManagerBase
	{
	public:
		/** �R���X�g���N�^ */
		LayerDLLManagerCPU()
		{
		}
		/** �f�X�g���N�^ */
		virtual ~LayerDLLManagerCPU()
		{
		}

	public:
		/** DLL��ǂݍ���ŁA�Ǘ��ɒǉ�����.
			@param szFilePath	�ǂݍ��ރt�@�C���̃p�X.
			@param o_addLayerCode	�ǉ����ꂽGUID�̊i�[��A�h���X.
			@return	���������ꍇ0���Ԃ�. */
		ErrorCode ReadLayerDLL(const wchar_t szFilePath[], Gravisbell::GUID& o_addLayerCode)
		{
			auto pLayerDLL = NNLayerDLL::CreateFromFileCPU(szFilePath, *this);
			if(pLayerDLL == NULL)
				return ERROR_CODE_DLL_LOAD_FUNCTION;

			Gravisbell::GUID guid;
			pLayerDLL->GetLayerCode(guid);

			// �Ǘ����C���[DLL���猟��
			auto pLayerDLLAlready = this->GetLayerDLLByGUID(guid);
			if(pLayerDLLAlready != NULL)
			{
				// ���ɒǉ��ς�
				delete pLayerDLL;
				return ERROR_CODE_DLL_ADD_ALREADY_SAMEID;
			}

			// �Ǘ��ɒǉ�
			this->lppNNLayerDLL.push_back(pLayerDLL);

			o_addLayerCode = guid;

			return ERROR_CODE_NONE;
		}
	};
	/** DLL�Ǘ��N���X(GPU����) */
	class LayerDLLManagerGPU : public LayerDLLManagerBase
	{
	public:
		/** �R���X�g���N�^ */
		LayerDLLManagerGPU()
		{
		}
		/** �f�X�g���N�^ */
		virtual ~LayerDLLManagerGPU()
		{
		}

	public:
		/** DLL��ǂݍ���ŁA�Ǘ��ɒǉ�����.
			@param szFilePath	�ǂݍ��ރt�@�C���̃p�X.
			@param o_addLayerCode	�ǉ����ꂽGUID�̊i�[��A�h���X.
			@return	���������ꍇ0���Ԃ�. */
		ErrorCode ReadLayerDLL(const wchar_t szFilePath[], Gravisbell::GUID& o_addLayerCode)
		{
			auto pLayerDLL = NNLayerDLL::CreateFromFileGPU(szFilePath, *this);
			if(pLayerDLL == NULL)
				return ERROR_CODE_DLL_LOAD_FUNCTION;

			Gravisbell::GUID guid;
			pLayerDLL->GetLayerCode(guid);

			// �Ǘ����C���[DLL���猟��
			auto pLayerDLLAlready = this->GetLayerDLLByGUID(guid);
			if(pLayerDLLAlready != NULL)
			{
				// ���ɒǉ��ς�
				delete pLayerDLL;
				return ERROR_CODE_DLL_ADD_ALREADY_SAMEID;
			}

			// �Ǘ��ɒǉ�
			this->lppNNLayerDLL.push_back(pLayerDLL);

			o_addLayerCode = guid;

			return ERROR_CODE_NONE;
		}
	};


	// DLL�Ǘ��N���X���쐬
	extern LayerDLLManager_API ILayerDLLManager* CreateLayerDLLManagerCPU()
	{
		return new LayerDLLManagerCPU();
	}
	// DLL�Ǘ��N���X���쐬
	extern LayerDLLManager_API ILayerDLLManager* CreateLayerDLLManagerGPU()
	{
		return new LayerDLLManagerGPU();
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

