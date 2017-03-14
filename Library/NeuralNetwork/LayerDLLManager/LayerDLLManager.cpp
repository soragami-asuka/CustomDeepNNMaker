// LayerDLLManager.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"
#include "Layer/NeuralNetwork/NNlayerFunction.h"
#include "LayerDLLManager.h"

#include<string>
#include<vector>
#include<rpc.h>

#pragma comment(lib, "Rpcrt4.lib")

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

		FuncCreateLayerConfig funcCreateLayerConfig;
		FuncCreateLayerConfigFromBuffer funcCreateLayerConfigFromBuffer;

		FuncCreateLayerCPU funcCreateLayerCPU;
		FuncCreateLayerGPU funcCreateLayerGPU;

	public:
		/** �R���X�g���N�^ */
		NNLayerDLL()
			:	hModule	(NULL)
			,	funcGetLayerCode				(NULL)
			,	funcGetVersionCode				(NULL)
			,	funcCreateLayerConfig			(NULL)
			,	funcCreateLayerConfigFromBuffer	(NULL)
			,	funcCreateLayerCPU				(NULL)
			,	funcCreateLayerGPU				(NULL)
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
		ErrorCode GetLayerCode(GUID& o_layerCode)const
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


		/** ���C���[�ݒ���쐬���� */
		SettingData::Standard::IData* CreateLayerConfig(void)const
		{
			if(this->funcCreateLayerConfig == NULL)
				return NULL;

			return this->funcCreateLayerConfig();
		}
		/** ���C���[�ݒ���쐬����
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@param o_useBufferSize ���ۂɓǂݍ��񂾃o�b�t�@�T�C�Y
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		SettingData::Standard::IData* CreateLayerConfigFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)const
		{
			if(this->funcCreateLayerConfigFromBuffer == NULL)
				return NULL;

			return this->funcCreateLayerConfigFromBuffer(i_lpBuffer, i_bufferSize, o_useBufferSize);
		}

		
		/** CPU�����p�̃��C���[���쐬.
			GUID�͎������蓖��. */
		INNLayer* CreateLayerCPU()const
		{
			UUID uuid;
			::UuidCreate(&uuid);

			return this->CreateLayerCPU(uuid);
		}
		/** CPU�����p�̃��C���[���쐬
			@param guid	�쐬���C���[��GUID */
		INNLayer* CreateLayerCPU(GUID guid)const
		{
			if(this->funcCreateLayerCPU == NULL)
				return NULL;

			return this->funcCreateLayerCPU(guid);
		}
		
		/** GPU�����p�̃��C���[���쐬.
			GUID�͎������蓖��. */
		INNLayer* CreateLayerGPU()const
		{
			UUID uuid;
			::UuidCreate(&uuid);

			return this->CreateLayerGPU(uuid);
		}
		/** GPU�����p�̃��C���[���쐬 */
		INNLayer* CreateLayerGPU(GUID guid)const
		{
			if(this->funcCreateLayerGPU == NULL)
				return NULL;

			return this->funcCreateLayerGPU(guid);
		}

	public:
		/** DLL���t�@�C������쐬���� */
		static NNLayerDLL* CreateFromFile(const ::std::wstring& filePath)
		{
			// �o�b�t�@���쐬
			NNLayerDLL* pLayerDLL = new NNLayerDLL();
			if(pLayerDLL == NULL)
				return NULL;

			do
			{
				// DLL�̓ǂݍ���
				pLayerDLL->hModule = LoadLibrary(filePath.c_str());
				if(pLayerDLL->hModule == NULL)
					break;

				// �֐��ǂݍ���
				pLayerDLL->funcGetLayerCode = (FuncGetLayerCode)GetProcAddress(pLayerDLL->hModule, "GetLayerCode");
				if(pLayerDLL->funcGetLayerCode == NULL)
					break;
				pLayerDLL->funcGetVersionCode = (FuncGetVersionCode)GetProcAddress(pLayerDLL->hModule, "GetVersionCode");
				if(pLayerDLL->funcGetVersionCode == NULL)
					break;

				pLayerDLL->funcCreateLayerConfig = (FuncCreateLayerConfig)GetProcAddress(pLayerDLL->hModule, "CreateLayerConfig");
				if(pLayerDLL->funcCreateLayerConfig == NULL)
					break;
				pLayerDLL->funcCreateLayerConfigFromBuffer = (FuncCreateLayerConfigFromBuffer)GetProcAddress(pLayerDLL->hModule, "CreateLayerConfigFromBuffer");
				if(pLayerDLL->funcCreateLayerConfigFromBuffer == NULL)
					break;

				pLayerDLL->funcCreateLayerCPU= (FuncCreateLayerCPU)GetProcAddress(pLayerDLL->hModule, "CreateLayerCPU");
				if(pLayerDLL->funcCreateLayerCPU == NULL)
					break;
				pLayerDLL->funcCreateLayerGPU= (FuncCreateLayerGPU)GetProcAddress(pLayerDLL->hModule, "CreateLayerGPU");
				if(pLayerDLL->funcCreateLayerGPU == NULL)
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
	class LayerDLLManager: public ILayerDLLManager
	{
	private:
		std::vector<NNLayerDLL*> lppNNLayerDLL;

	public:
		/** �R���X�g���N�^ */
		LayerDLLManager()
		{
		}
		/** �f�X�g���N�^ */
		virtual ~LayerDLLManager()
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
		ErrorCode ReadLayerDLL(const wchar_t szFilePath[], GUID& o_addLayerCode)
		{
			auto pLayerDLL = NNLayerDLL::CreateFromFile(szFilePath);
			if(pLayerDLL == NULL)
				return ERROR_CODE_DLL_LOAD_FUNCTION;

			GUID guid;
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
		/** DLL��ǂݍ���ŁA�Ǘ��ɒǉ�����.
			@param szFilePath	�ǂݍ��ރt�@�C���̃p�X.
			@return	���������ꍇ0���Ԃ�. */
		ErrorCode ReadLayerDLL(const wchar_t szFilePath[])
		{
			GUID layerCode;
			return this->ReadLayerDLL(szFilePath, layerCode);
		}

		/** �Ǘ����Ă��郌�C���[DLL�̐����擾���� */
		unsigned int GetLayerDLLCount()const
		{
			return this->lppNNLayerDLL.size();
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
		const ILayerDLL* GetLayerDLLByGUID(GUID i_layerCode)const
		{
			for(unsigned int i=0; i<this->lppNNLayerDLL.size(); i++)
			{
				auto pLayerDLL = this->lppNNLayerDLL[i];
				if(pLayerDLL == NULL)
					continue;

				// GUID���擾����
				GUID layerCode;
				if(pLayerDLL->GetLayerCode(layerCode) != 0)
					continue;

				// �m�F
				if(layerCode == i_layerCode)
					return pLayerDLL;
			}

			return NULL;
		}

		/** ���C���[DLL���폜����. */
		ErrorCode EraseLayerDLL(GUID i_layerCode)
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
				GUID layerCode;
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


	// DLL�Ǘ��N���X���쐬
	extern LayerDLLManager_API ILayerDLLManager* CreateLayerDLLManager()
	{
		return new LayerDLLManager();
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

