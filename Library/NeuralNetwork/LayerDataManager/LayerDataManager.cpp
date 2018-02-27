//=========================================
// �j���[�����l�b�g���[�N�p���C���[�f�[�^�̊Ǘ��N���X
//=========================================
#include"stdafx.h"

#include<map>

#include"Library/NeuralNetwork/LayerDataManager.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class LayerDataManager : public ILayerDataManager
	{
	private:
		std::map<Gravisbell::GUID, ILayerData*> lpLayerData;

	public:
		LayerDataManager()
			:	ILayerDataManager()
		{
		}
		virtual ~LayerDataManager()
		{
			this->ClearLayerData();
		}

	public:
		/** ���C���[�f�[�^�̍쐬.�����I�ɊǗ��܂ōs��.
			@param	i_layerDLLManager	���C���[DLL�Ǘ��N���X.
			@param	i_typeCode			���C���[��ʃR�[�h
			@param	i_guid				�V�K�쐬���郌�C���[�f�[�^��GUID
			@param	i_layerStructure	���C���[�\��
			@param	i_inputDataStruct	���̓f�[�^�\��
			@param	o_pErrorCode		�G���[�R�[�h�i�[��̃A�h���X. NULL�w���.
			@return
			typeCode�����݂��Ȃ��ꍇ�ANULL��Ԃ�.
			���ɑ��݂���guid��typeCode����v�����ꍇ�A�����ۗL�̃��C���[�f�[�^��Ԃ�.
			���ɑ��݂���guid��typeCode���قȂ�ꍇ�ANULL��Ԃ�. */
		ILayerData* CreateLayerData(
			const ILayerDLLManager& i_layerDLLManager,
			const Gravisbell::GUID& i_typeCode,
			const Gravisbell::GUID& i_guid,
			const SettingData::Standard::IData& i_layerStructure,
			Gravisbell::ErrorCode* o_pErrorCode = NULL)
		{
			// ���ꃌ�C���[�����݂��Ȃ����m�F
			if(this->lpLayerData.count(i_guid))
			{
				if(lpLayerData[i_guid]->GetLayerCode() == i_typeCode)
				{
					return lpLayerData[i_guid];
				}
				else
				{
					if(o_pErrorCode)
						*o_pErrorCode = ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;
					return NULL;
				}
			}

			// DLL������
			auto pLayerDLL = i_layerDLLManager.GetLayerDLLByGUID(i_typeCode);
			if(pLayerDLL == NULL)
			{
				if(o_pErrorCode)
					*o_pErrorCode = ErrorCode::ERROR_CODE_DLL_NOTFOUND;
				return NULL;
			}

			// ���C���[�f�[�^���쐬
			auto pLayerData = pLayerDLL->CreateLayerData(i_guid, i_layerStructure);
			if(pLayerData == NULL)
			{
				if(o_pErrorCode)
					*o_pErrorCode = ErrorCode::ERROR_CODE_LAYER_CREATE;
				return NULL;
			}

			// ���C���[�f�[�^��ۑ�
			this->lpLayerData[i_guid] = pLayerData;

			if(o_pErrorCode)
				*o_pErrorCode = ErrorCode::ERROR_CODE_NONE;

			return pLayerData;
		}



		/** ���C���[�f�[�^���o�b�t�@����쐬.�����I�ɊǗ��܂ōs��.
			@param	i_layerDLLManager	���C���[DLL�Ǘ��N���X.
			@param	i_typeCode			���C���[��ʃR�[�h
			@param	i_guid				�V�K�쐬���郌�C���[�f�[�^��GUID
			@param	i_lpBuffer		�ǂݎ��p�o�b�t�@.
			@param	i_bufferSize	�g�p�\�ȃo�b�t�@�T�C�Y.
			@param	o_useBufferSize	���ۂɎg�p�����o�b�t�@�T�C�Y.
			@param	o_pErrorCode		�G���[�R�[�h�i�[��̃A�h���X. NULL�w���.
			@return
			typeCode�����݂��Ȃ��ꍇ�ANULL��Ԃ�.
			���ɑ��݂���guid��typeCode����v�����ꍇ�A�����ۗL�̃��C���[�f�[�^��Ԃ�.
			���ɑ��݂���guid��typeCode���قȂ�ꍇ�ANULL��Ԃ�. */
		ILayerData* CreateLayerData(
			const ILayerDLLManager& i_layerDLLManager,
			const Gravisbell::GUID& i_typeCode,
			const Gravisbell::GUID& i_guid,
			const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize,
			Gravisbell::ErrorCode* o_pErrorCode = NULL)
		{
			// DLL������
			auto pLayerDLL = i_layerDLLManager.GetLayerDLLByGUID(i_typeCode);
			if(pLayerDLL == NULL)
			{
				if(o_pErrorCode)
					*o_pErrorCode = ErrorCode::ERROR_CODE_DLL_NOTFOUND;
				return NULL;
			}

			// ���C���[�f�[�^���쐬
			auto pLayerData = pLayerDLL->CreateLayerDataFromBuffer(i_guid, i_lpBuffer, i_bufferSize, o_useBufferSize);
			if(pLayerData == NULL)
			{
				if(o_pErrorCode)
					*o_pErrorCode = ErrorCode::ERROR_CODE_LAYER_CREATE;
				return NULL;
			}


			// ���ꃌ�C���[�����݂��Ȃ����m�F
			if(this->lpLayerData.count(i_guid))
			{
				delete pLayerData;
				if(lpLayerData[i_guid]->GetLayerCode() == i_typeCode)
				{
					return lpLayerData[i_guid];
				}
				else
				{
					if(o_pErrorCode)
						*o_pErrorCode = ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;
					return NULL;
				}
			}

			// ���C���[�f�[�^��ۑ�
			this->lpLayerData[i_guid] = pLayerData;

			if(o_pErrorCode)
				*o_pErrorCode = ErrorCode::ERROR_CODE_NONE;

			return pLayerData;
		}


		/** ���C���[�f�[�^��GUID�w��Ŏ擾���� */
		ILayerData* GetLayerData(const Gravisbell::GUID& i_guid)
		{
			if(this->lpLayerData.count(i_guid))
				return this->lpLayerData[i_guid];
			return NULL;
		}

		/** ���C���[�f�[�^�����擾���� */
		U32 GetLayerDataCount()
		{
			return (U32)this->lpLayerData.size();
		}
		/** ���C���[�f�[�^��ԍ��w��Ŏ擾���� */
		ILayerData* GetLayerDataByNum(U32 i_num)
		{
			if(i_num >= this->lpLayerData.size())
				return NULL;

			auto it = this->lpLayerData.begin();
			for(U32 i=0; i<i_num; i++)
				it++;

			return it->second;
		}

		/** ���C���[�f�[�^��GUID�w��ō폜���� */
		Gravisbell::ErrorCode EraseLayerByGUID(const Gravisbell::GUID& i_guid)
		{
			auto it = this->lpLayerData.find(i_guid);
			if(it == this->lpLayerData.end())
				return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

			delete it->second;
			this->lpLayerData.erase(it);

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** ���C���[�f�[�^��ԍ��w��ō폜���� */
		Gravisbell::ErrorCode EraseLayerByNum(U32 i_num)
		{
			if(i_num >= this->lpLayerData.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			auto it = this->lpLayerData.begin();
			for(U32 i=0; i<i_num; i++)
				it++;

			delete it->second;
			this->lpLayerData.erase(it);

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** ���C���[�f�[�^�����ׂč폜���� */
		Gravisbell::ErrorCode ClearLayerData()
		{
			auto it = this->lpLayerData.begin();
			while(it != this->lpLayerData.end())
			{
				delete it->second;
				it = this->lpLayerData.erase(it);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** ���C���[�f�[�^�̊Ǘ��N���X���쐬 */
	LayerDataManager_API ILayerDataManager* CreateLayerDataManager()
	{
		return new LayerDataManager();
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell