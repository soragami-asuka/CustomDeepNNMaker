//=======================================
// ���C���[�Ǘ��N���X
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_MANAGER_H__
#define __GRAVISBELL_I_NN_LAYER_MANAGER_H__

#include"../../Common/Guiddef.h"
#include"../../Common/ErrorCode.h"

#include"ILayerDLLManager.h"
#include"INNLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class ILayerDataManager
	{
	public:
		/** �R���X�g���N�^ */
		ILayerDataManager(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerDataManager(){}

	public:
		/** ���C���[�f�[�^�̍쐬.	�����I�ɊǗ��܂ōs��.
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
		virtual ILayerData* CreateLayerData(
			const ILayerDLLManager& i_layerDLLManager, const Gravisbell::GUID& i_typeCode,
			const Gravisbell::GUID& i_guid,
			const SettingData::Standard::IData& i_layerStructure,
			const IODataStruct& i_inputDataStruct,
			Gravisbell::ErrorCode* o_pErrorCode = NULL) = 0;
		
		/** ���C���[�f�[�^���o�b�t�@����쐬.�����I�ɊǗ��܂ōs��.
			@param	i_layerDLLManager	���C���[DLL�Ǘ��N���X.
			@param	i_typeCode			���C���[��ʃR�[�h
			@param	i_guid				�V�K�쐬���郌�C���[�f�[�^��GUID
			@param	i_lpBuffer			�ǂݎ��p�o�b�t�@.
			@param	i_bufferSize		�g�p�\�ȃo�b�t�@�T�C�Y.
			@param	o_useBufferSize		���ۂɎg�p�����o�b�t�@�T�C�Y.
			@param	o_pErrorCode		�G���[�R�[�h�i�[��̃A�h���X. NULL�w���.
			@return
			typeCode�����݂��Ȃ��ꍇ�ANULL��Ԃ�.
			���ɑ��݂���guid��typeCode����v�����ꍇ�A�����ۗL�̃��C���[�f�[�^��Ԃ�.
			���ɑ��݂���guid��typeCode���قȂ�ꍇ�ANULL��Ԃ�. */
		virtual ILayerData* CreateLayerData(
			const ILayerDLLManager& i_layerDLLManager,
			const Gravisbell::GUID& i_typeCode,
			const Gravisbell::GUID& i_guid,
			const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize,
			Gravisbell::ErrorCode* o_pErrorCode = NULL) = 0;


		/** ���C���[�f�[�^��GUID�w��Ŏ擾���� */
		virtual ILayerData* GetLayerData(const Gravisbell::GUID& i_guid) = 0;

		/** ���C���[�f�[�^�����擾���� */
		virtual U32 GetLayerDataCount() = 0;
		/** ���C���[�f�[�^��ԍ��w��Ŏ擾���� */
		virtual ILayerData* GetLayerDataByNum(U32 i_num) = 0;

		/** ���C���[�f�[�^��GUID�w��ō폜���� */
		virtual Gravisbell::ErrorCode EraseLayerByGUID(const Gravisbell::GUID& i_guid) = 0;
		/** ���C���[�f�[�^��ԍ��w��ō폜���� */
		virtual Gravisbell::ErrorCode EraseLayerByNum(U32 i_num) = 0;

		/** ���C���[�f�[�^�����ׂč폜���� */
		virtual Gravisbell::ErrorCode ClearLayerData() = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif