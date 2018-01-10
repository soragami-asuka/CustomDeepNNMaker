//=======================================
// �j���[�����l�b�g���[�N�̃��C���[�Ɋւ���f�[�^����舵���C���^�[�t�F�[�X
// �o�b�t�@�Ȃǂ��Ǘ�����.
//=======================================
#ifndef __GRAVISBELL_I_LAYER_CONNECT_DATA_H__
#define __GRAVISBELL_I_LAYER_CONNECT_DATA_H__

#include"../ILayerData.h"
#include"../NeuralNetwork/INeuralNetwork.h"

namespace Gravisbell {
namespace Layer {
namespace Connect {

	class ILayerConnectData : public ILayerData
	{
		//====================================
		// �R���X�g���N�^/�f�X�g���N�^
		//====================================
	public:
		/** �R���X�g���N�^ */
		ILayerConnectData() : ILayerData(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerConnectData(){}

		//===========================
		// ���C���[�\��
		//===========================
		/** ���̓f�[�^�\�����g�p�\���m�F����.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@return	�g�p�\�ȓ��̓f�[�^�\���̏ꍇtrue���Ԃ�. */
		virtual bool CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;
		virtual bool CheckCanUseInputDataStruct(Gravisbell::GUID i_guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;

		/** �o�̓f�[�^�\�����擾����.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
		virtual IODataStruct GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;
		virtual IODataStruct GetOutputDataStruct(Gravisbell::GUID i_guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;


		//====================================
		// ���C���[�̒ǉ�/�폜/�Ǘ�
		//====================================
	public:
		/** ���C���[�f�[�^��ǉ�����.
			@param	i_guid			�ǉ����郌�C���[�Ɋ��蓖�Ă���GUID.
			@param	i_pLayerData	�ǉ����郌�C���[�f�[�^�̃A�h���X.
			@param	i_onFixFlag		���C���[���Œ艻����t���O. */
		virtual ErrorCode AddLayer(const Gravisbell::GUID& i_guid, ILayerData* i_pLayerData, bool i_onFixFlag) = 0;
		/** ���C���[�f�[�^���폜����.
			@param i_guid	�폜���郌�C���[��GUID */
		virtual ErrorCode EraseLayer(const Gravisbell::GUID& i_guid) = 0;
		/** ���C���[�f�[�^��S�폜���� */
		virtual ErrorCode EraseAllLayer() = 0;

		/** �o�^����Ă��郌�C���[�����擾���� */
		virtual U32 GetLayerCount() = 0;
		/** ���C���[��GUID��ԍ��w��Ŏ擾���� */
		virtual ErrorCode GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid) = 0;

		/** �o�^����Ă��郌�C���[�f�[�^��ԍ��w��Ŏ擾���� */
		virtual ILayerData* GetLayerDataByNum(U32 i_layerNum) = 0;
		/** �o�^����Ă��郌�C���[�f�[�^��GUID�w��Ŏ擾���� */
		virtual ILayerData* GetLayerDataByGUID(const Gravisbell::GUID& i_guid) = 0;

		/** ���C���[�̌Œ艻�t���O���擾���� */
		virtual bool GetLayerFixFlagByGUID(const Gravisbell::GUID& i_guid) = 0;

		//====================================
		// ���o�̓��C���[
		//====================================
	public:
		/** ���͐M���Ɋ��蓖�Ă��Ă���GUID���擾���� */
		virtual GUID GetInputGUID() = 0;

		/** �o�͐M�����C���[��ݒ肷�� */
		virtual ErrorCode SetOutputLayerGUID(const Gravisbell::GUID& i_guid) = 0;
		/** �o�͐M�����C���[��GUID���擾���� */
		virtual Gravisbell::GUID GetOutputLayerGUID() = 0;


		//====================================
		// ���C���[�̐ڑ�
		//====================================
	public:
		/** ���C���[�ɓ��̓��C���[��ǉ�����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		virtual ErrorCode AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;
		/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		virtual ErrorCode AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;

		/** ���C���[������̓��C���[���폜����. 
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		virtual ErrorCode EraseInputLayerFromLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;
		/** ���C���[����o�C�p�X���C���[���폜����.
			@param	receiveLayer	���͂��󂯎�郌�C���[
			@param	postLayer		���͂�n��(�o�͂���)���C���[. */
		virtual ErrorCode EraseBypassLayerFromLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer) = 0;

		/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		virtual ErrorCode ResetInputLayer(const Gravisbell::GUID& layerGUID) = 0;
		/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
			@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
		virtual ErrorCode ResetBypassLayer(const Gravisbell::GUID& layerGUID) = 0;

		/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		virtual U32 GetInputLayerCount(const Gravisbell::GUID& i_layerGUID) = 0;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
		virtual U32 GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID) = 0;

		/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
			@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
		virtual ErrorCode GetInputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID) = 0;
		/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
			@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID.
			@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��.
			@param	o_postLayerGUID	���C���[�ɐڑ����Ă��郌�C���[��GUID�i�[��. */
		virtual ErrorCode GetBypassLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID) = 0;

	public:
		//===========================
		// ���C���[�쐬
		//===========================
		/** ���C���[���쐬����.
			@param	guid	�V�K�������郌�C���[��GUID.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager) = 0;
		/** ���C���[���쐬����.
			@param	guid	�V�K�������郌�C���[��GUID.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@param	i_useHostMemory		�����I�Ƀz�X�g���������g�p����. */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager, bool i_useHostMemory) = 0;

		/** ���C���[���쐬����.
			@param	guid	�V�K�������郌�C���[��GUID.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount) = 0;
		/** ���C���[���쐬����.
			@param	guid	�V�K�������郌�C���[��GUID.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@param	i_useHostMemory		�����I�Ƀz�X�g���������g�p����. */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, bool i_useHostMemory) = 0;
	};

}	// Connect
}	// Layer
}	// Gravisbell

#endif